#!/usr/bin/env python3
"""Run realtime YOLO detection on a RealSense raw image topic."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO


DEFAULT_WEIGHTS = (
    "/home/skuba/skuba_ws/src/skuba_vision_project/datagen/"
    "evaluation/train_best_tuned/weights/best.pt"
)
DEFAULT_TOPIC = "/camera/camera/color/image_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects from a RealSense raw image topic in realtime."
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Path to YOLO .pt weights.",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Raw sensor_msgs/Image topic from the RealSense camera.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example '0', 'cuda:0', or 'cpu'. Default lets YOLO choose.",
    )
    parser.add_argument(
        "--window-name",
        default="RealSense YOLO Detection",
        help="OpenCV display window name.",
    )
    return parser.parse_known_args()[0]


class RealtimeDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        weights = Path(args.weights).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        self.args = args
        self.bridge = CvBridge()
        self.model = YOLO(str(weights))
        self.last_frame_time = time.monotonic()

        cv2.namedWindow(self.args.window_name, cv2.WINDOW_NORMAL)

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            print(f"Failed to convert ROS image: {exc}", file=sys.stderr)
            return

        predict_kwargs = {
            "source": frame,
            "conf": self.args.conf,
            "imgsz": self.args.imgsz,
            "verbose": False,
        }
        if self.args.device:
            predict_kwargs["device"] = self.args.device

        results = self.model.predict(**predict_kwargs)
        annotated = results[0].plot()

        now = time.monotonic()
        fps = 1.0 / max(now - self.last_frame_time, 1e-6)
        self.last_frame_time = now
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(self.args.window_name, annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            self.request_shutdown()

    def request_shutdown(self) -> None:
        pass

    def close(self) -> None:
        cv2.destroyAllWindows()


def run_ros2(args: argparse.Namespace) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    class YoloImageNode(Node):
        def __init__(self) -> None:
            super().__init__("model_test_yolo_detector")
            self.detector = RealtimeDetector(args)
            self.detector.request_shutdown = self.shutdown

            qos = QoSProfile(depth=1)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.subscription = self.create_subscription(
                Image,
                args.topic,
                self.detector.image_callback,
                qos,
            )
            self.get_logger().info(f"Listening to image topic: {args.topic}")
            self.get_logger().info(f"Using YOLO weights: {args.weights}")

        def shutdown(self) -> None:
            rclpy.shutdown()

        def destroy_node(self) -> bool:
            self.detector.close()
            return super().destroy_node()

    rclpy.init(args=None)
    node = YoloImageNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_ros1(args: argparse.Namespace) -> None:
    import rospy

    detector = RealtimeDetector(args)
    detector.request_shutdown = lambda: rospy.signal_shutdown("display window closed")

    rospy.init_node("model_test_yolo_detector", anonymous=True)
    rospy.loginfo("Listening to image topic: %s", args.topic)
    rospy.loginfo("Using YOLO weights: %s", args.weights)
    rospy.Subscriber(
        args.topic,
        Image,
        detector.image_callback,
        queue_size=1,
        buff_size=2**24,
    )

    try:
        rospy.spin()
    finally:
        detector.close()


def main() -> None:
    args = parse_args()

    try:
        import rclpy  # noqa: F401
    except ImportError:
        try:
            import rospy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Neither rclpy nor rospy is available. Source your ROS environment first."
            ) from exc
        run_ros1(args)
    else:
        run_ros2(args)


if __name__ == "__main__":
    main()
