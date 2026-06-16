#!/usr/bin/env python3
"""
Overlay bounding boxes onto raw video using ROS bag detections.
Syncs by frame_id to ensure perfect alignment.
"""

import cv2
import sys
import os
from pathlib import Path
from collections import defaultdict

# ROS2 imports
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
    from rclpy.serialization import deserialize_message
except ImportError:
    print("Error: ROS2 Python libraries not found.")
    print("Install: sudo apt install ros-humble-rosbag2-py python3-rosbag2")
    sys.exit(1)


def read_detections_from_bag(bag_path):
    """Read all detections from ROS bag, indexed by frame_id."""
    detections_by_frame = defaultdict(list)
    
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get message type
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    msg_type = get_message(type_map['/drone/detections'])
    
    print(f"Reading detections from {bag_path}...")
    count = 0
    
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        if topic == '/drone/detections':
            msg = deserialize_message(data, msg_type)
            
            # Parse CSV format: "frame_id,class_id,confidence,x,y,width,height\n..."
            for line in msg.data.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) == 7:
                        frame_id = int(parts[0])
                        detection = {
                            'class_id': int(parts[1]),
                            'confidence': float(parts[2]),
                            'x': int(parts[3]),
                            'y': int(parts[4]),
                            'width': int(parts[5]),
                            'height': int(parts[6])
                        }
                        detections_by_frame[frame_id].append(detection)
                        count += 1
    
    print(f"Loaded {count} detections across {len(detections_by_frame)} frames")
    return detections_by_frame


def overlay_video(frames_path, detections_by_frame, output_path, processed_width, processed_height):
    """Read frames and overlay bounding boxes frame by frame."""
    
    # Check if input is a video file or directory of images
    if frames_path.is_dir():
        # Directory of images
        frame_files = sorted(frames_path.glob("frame_*.jpg"))
        if not frame_files:
            print(f"Error: No frame_*.jpg files found in {frames_path}")
            return False
        
        print(f"\nFound {len(frame_files)} frames in directory")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            print(f"Error: Cannot read first frame")
            return False
        
        height, width = first_frame.shape[:2]
        fps = 30  # Default FPS for image sequences
        total_frames = len(frame_files)
        
        # Calculate scale factors
        scale_x = width / processed_width
        scale_y = height / processed_height
        
        print(f"Frame resolution: {width}x{height}")
        print(f"Processed resolution: {processed_width}x{processed_height}")
        print(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Creating annotated video: {output_path}")
        print("Processing frames...")
        
        frame_id = 0
        frames_with_detections = 0
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not read {frame_file}")
                continue
            
            # Get detections for this frame
            if frame_id in detections_by_frame:
                detections = detections_by_frame[frame_id]
                frames_with_detections += 1
                
                # Draw each detection
                for det in detections:
                    # Scale bbox coordinates from processed resolution to original resolution
                    x = int(det['x'] * scale_x)
                    y = int(det['y'] * scale_y)
                    w = int(det['width'] * scale_x)
                    h = int(det['height'] * scale_y)
                    conf = det['confidence']
                    class_id = det['class_id']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw label with frame ID
                    label = f"F{frame_id} ID:{class_id} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Background for text
                    cv2.rectangle(frame, (x, y-label_size[1]-5), 
                                (x+label_size[0], y), (0, 255, 0), -1)
                    
                    # Text
                    cv2.putText(frame, label, (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add frame number in corner
            cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            
            # Progress indicator
            if frame_id % 30 == 0:
                progress = (frame_id / total_frames) * 100
                print(f"  Frame {frame_id}/{total_frames} ({progress:.1f}%)")
            
            frame_id += 1
        
        out.release()
        
    else:
        # Video file (legacy support)
        cap = cv2.VideoCapture(str(frames_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {frames_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate scale factors (bbox coords are in processed resolution, need to scale to original)
        scale_x = width / processed_width
        scale_y = height / processed_height
        
        print(f"\nVideo: {width}x{height} @ {fps} FPS, {total_frames} frames")
        print(f"Processed: {processed_width}x{processed_height}")
        print(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Creating annotated video: {output_path}")
        print("Processing frames...")
        
        frame_id = 0
        frames_with_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections for this frame
            if frame_id in detections_by_frame:
                detections = detections_by_frame[frame_id]
                frames_with_detections += 1
                
                # Draw each detection
                for det in detections:
                    # Scale bbox coordinates from processed resolution to original video resolution
                    x = int(det['x'] * scale_x)
                    y = int(det['y'] * scale_y)
                    w = int(det['width'] * scale_x)
                    h = int(det['height'] * scale_y)
                    conf = det['confidence']
                    class_id = det['class_id']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw label with frame ID
                    label = f"F{frame_id} ID:{class_id} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Background for text
                    cv2.rectangle(frame, (x, y-label_size[1]-5), 
                                (x+label_size[0], y), (0, 255, 0), -1)
                    
                    # Text
                    cv2.putText(frame, label, (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add frame number in corner
            cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            
            # Progress indicator
            if frame_id % 30 == 0:
                progress = (frame_id / total_frames) * 100
                print(f"  Frame {frame_id}/{total_frames} ({progress:.1f}%)")
            
            frame_id += 1
        
        cap.release()
        out.release()
    
    print(f"\nâœ“ Done! Processed {frame_id} frames")
    print(f"âœ“ {frames_with_detections} frames had detections")
    print(f"âœ“ Output: {output_path}")
    
    return True


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 overlay_detections.py <flight_name> <processed_width> <processed_height>")
        print("")
        print("Example:")
        print("  python3 overlay_detections.py test_flight 640 360")
        print("")
        print("Where 640x360 is the resolution used by the detector (--width and --height flags)")
        print("")
        print("This will read:")
        print("  ~/yolo_test/test_flight/           (ROS bag)")
        print("  ~/yolo_test/test_flight_frames/    (frames directory)")
        print("  OR ~/yolo_test/test_flight_raw.mp4 (video file, legacy)")
        print("")
        print("And create:")
        print("  ~/yolo_test/test_flight_annotated.mp4")
        sys.exit(1)
    
    flight_name = sys.argv[1]
    processed_width = int(sys.argv[2])
    processed_height = int(sys.argv[3])
    base_dir = Path(os.environ.get("YOLO_OUTPUT_DIR", Path.home() / "yolo_test"))
    
    bag_path = base_dir / flight_name
    frames_path = base_dir / f"{flight_name}_frames"
    video_path = base_dir / f"{flight_name}_raw.mp4"
    output_path = base_dir / f"{flight_name}_annotated.mp4"
    
    # Validate inputs
    if not bag_path.exists():
        print(f"Error: ROS bag not found: {bag_path}")
        sys.exit(1)
    
    # Check for frames directory first, then video file
    if frames_path.exists() and frames_path.is_dir():
        input_path = frames_path
        input_type = "frames directory"
    elif video_path.exists():
        input_path = video_path
        input_type = "video file"
    else:
        print(f"Error: Neither frames directory nor video found:")
        print(f"  {frames_path}")
        print(f"  {video_path}")
        sys.exit(1)
    
    print("=" * 50)
    print("  Detection Overlay Tool")
    print("=" * 50)
    print(f"Flight: {flight_name}")
    print(f"Processed resolution: {processed_width}x{processed_height}")
    print(f"Bag:    {bag_path}")
    print(f"Input:  {input_path} ({input_type})")
    print(f"Output: {output_path}")
    print("=" * 50)
    
    # Step 1: Read all detections from bag
    detections_by_frame = read_detections_from_bag(bag_path)
    
    if not detections_by_frame:
        print("\nWarning: No detections found in bag!")
        print("The output video will be the same as input.")
    
    # Step 2: Process frames/video and overlay detections
    success = overlay_video(input_path, detections_by_frame, output_path, processed_width, processed_height)
    
    if success:
        print("\n" + "=" * 50)
        print("  View annotated video:")
        print(f"  vlc {output_path}")
        print("=" * 50)
    else:
        print("\nError: Failed to create annotated video")
        sys.exit(1)


if __name__ == "__main__":
    main()
