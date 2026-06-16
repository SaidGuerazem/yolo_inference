#!/usr/bin/env python3
"""
Visualize detections on raw camera images from ROS bag.
Usage: python3 visualize_detections.py ~/yolo_test/flight_1
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import argparse
import os
from collections import deque

class DetectionVisualizer(Node):
    def __init__(self, output_dir):
        super().__init__('detection_visualizer')
        self.bridge = CvBridge()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/drone/raw_camera', self.image_callback, 10)
        self.detection_sub = self.create_subscription(
            String, '/drone/detections', self.detection_callback, 10)
        
        self.latest_detections = None
        self.frame_count = 0
        self.get_logger().info(f'Saving visualizations to: {output_dir}')
        
    def detection_callback(self, msg):
        # Parse detections: "frame_id,class_id,confidence,x,y,width,height\n..."
        detections = []
        for line in msg.data.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) == 7:
                    frame_id, class_id, conf, x, y, w, h = parts
                    detections.append({
                        'frame_id': int(frame_id),
                        'class_id': int(class_id),
                        'confidence': float(conf),
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    })
        self.latest_detections = detections
        
    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Draw detections if available
        if self.latest_detections:
            for det in self.latest_detections:
                x, y, w, h = det['x'], det['y'], det['width'], det['height']
                conf = det['confidence']
                class_id = det['class_id']
                frame_id = det['frame_id']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw label with frame ID
                label = f"F{frame_id} ID:{class_id} {conf:.2f}"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save frame
        output_path = os.path.join(self.output_dir, f'frame_{self.frame_count:06d}.jpg')
        cv2.imwrite(output_path, frame)
        self.frame_count += 1
        
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'Processed {self.frame_count} frames')

def main():
    parser = argparse.ArgumentParser(description='Visualize detections on raw images')
    parser.add_argument('bag_path', help='Path to ROS bag directory')
    parser.add_argument('--output', default='visualized_frames', 
                       help='Output directory for visualized frames')
    args = parser.parse_args()
    
    rclpy.init()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(args.bag_path), args.output)
    node = DetectionVisualizer(output_dir)
    
    print(f"\nPlaying bag: {args.bag_path}")
    print(f"Saving to: {output_dir}")
    print("\nIn another terminal, run:")
    print(f"  ros2 bag play {args.bag_path}\n")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print(f"\nVisualization complete! Frames saved to: {output_dir}")

if __name__ == '__main__':
    main()
