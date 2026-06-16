#!/usr/bin/env python3
"""
Publish the dual MIPI/CSI camera composite to /drone/raw_camera.
Usage: python3 record_raw_camera.py
"""

import argparse
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class RawCameraPublisher(Node):
    def __init__(self, width, height, fps, sensor0, sensor1,
                 sensor_width, sensor_height, sensor_fps):
        super().__init__('raw_camera_publisher')
        self.publisher = self.create_publisher(Image, '/drone/raw_camera', 10)
        self.bridge = CvBridge()

        sink_width = width // 2
        pipeline = (
            f"nvcompositor name=comp "
            f"sink_0::width={sink_width} sink_0::height={height} "
            f"sink_0::xpos=0 sink_0::ypos=0 "
            f"sink_1::width={sink_width} sink_1::height={height} "
            f"sink_1::xpos={sink_width} sink_1::ypos=0 ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
            f"queue ! nvvidconv interpolation-method=1 ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1 "
            f"nvarguscamerasrc sensor-id={sensor0} gainrange=\"3 10\" exposuretimerange=\"40000 8333000\" ! "
            f"video/x-raw(memory:NVMM), width={sensor_width}, height={sensor_height}, "
            f"framerate={sensor_fps}/1, format=NV12 ! "
            f"videorate ! video/x-raw(memory:NVMM), framerate={fps}/1, format=NV12 ! comp.sink_0 "
            f"nvarguscamerasrc sensor-id={sensor1} gainrange=\"3 10\" exposuretimerange=\"40000 8333000\" ! "
            f"video/x-raw(memory:NVMM), width={sensor_width}, height={sensor_height}, "
            f"framerate={sensor_fps}/1, format=NV12 ! "
            f"videorate ! video/x-raw(memory:NVMM), framerate={fps}/1, format=NV12 ! comp.sink_1"
        )

        self.get_logger().info(f'Opening MIPI cameras: {pipeline}')
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open MIPI cameras')
            sys.exit(1)

        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.frame_count = 0

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mipi_camera_frame'
        self.publisher.publish(msg)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'Published {self.frame_count} frames')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description='Publish dual MIPI camera feed to ROS')
    parser.add_argument('--width', type=int, default=3840, help='Composited frame width')
    parser.add_argument('--height', type=int, default=1200, help='Composited frame height')
    parser.add_argument('--fps', type=int, default=20, help='Output frame rate')
    parser.add_argument('--sensor0', type=int, default=0, help='First MIPI/CSI sensor id')
    parser.add_argument('--sensor1', type=int, default=1, help='Second MIPI/CSI sensor id')
    parser.add_argument('--sensor-width', type=int, default=1920,
                        help='Per-sensor capture width')
    parser.add_argument('--sensor-height', type=int, default=1200,
                        help='Per-sensor capture height')
    parser.add_argument('--sensor-fps', type=int, default=60,
                        help='Per-sensor capture FPS before videorate')
    args = parser.parse_args()

    rclpy.init()
    node = RawCameraPublisher(
        args.width, args.height, args.fps,
        args.sensor0, args.sensor1,
        args.sensor_width, args.sensor_height,
        args.sensor_fps,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
