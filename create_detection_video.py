import os
import cv2
import argparse
import re

def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)\.jpg', filename)
    return int(match.group(1)) if match else 0

def create_video_from_dir(directory, fps=20):
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return False

    # Find and sort JPEGs numerically
    jpegs = [f for f in os.listdir(directory) if f.endswith(".jpg") and f.startswith("frame_")]
    if not jpegs:
        print(f"No frames found in {directory}.")
        return False

    jpegs.sort(key=extract_frame_number)
    print(f"Processing {len(jpegs)} frames in {directory}...")

    # Read first image to get dimensions
    first_img_path = os.path.join(directory, jpegs[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Error: Could not read first frame {first_img_path}")
        return False

    h, w, _ = first_img.shape
    
    # Set output video path
    dir_name = os.path.basename(os.path.normpath(directory))
    parent_dir = os.path.dirname(os.path.normpath(directory))
    output_path = os.path.join(parent_dir, f"{dir_name}_annotated.mp4")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for filename in jpegs:
        img_path = os.path.join(directory, filename)
        txt_path = img_path.replace(".jpg", ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Check for matching YOLO label file
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                box_w = float(parts[3]) * w
                box_h = float(parts[4]) * h

                # Read confidence if present (6th column), default to 1.0
                confidence = float(parts[5]) if len(parts) >= 6 else 1.0

                # Determine bounding box color
                # BGR format: Green (0,255,0), Orange (0,165,255), Red (0,0,255)
                if confidence >= 0.70:
                    color = (0, 255, 0)       # Green
                elif confidence >= 0.50:
                    color = (0, 165, 255)     # Orange
                else:
                    color = (0, 0, 255)       # Red

                xmin = int(cx - box_w / 2.0)
                ymin = int(cy - box_h / 2.0)
                xmax = int(cx + box_w / 2.0)
                ymax = int(cy + box_h / 2.0)

                # Draw bounding box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                # Draw label
                label = f"Class {class_id} ({confidence*100:.1f}%)"
                
                # Background text block
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (xmin, ymin - label_size[1] - 5), (xmin + label_size[0], ymin), color, -1)
                
                # Write text
                cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Write frame to video
        out.write(img)

    out.release()
    print(f"??? Video successfully created: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create annotated MP4 video from saved session folders.")
    parser.add_argument("session_dir", type=str, help="Path to the session directory (e.g. /home/sapience/ros_pkgs/saved_images/session_YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    if not os.path.exists(args.session_dir):
        print(f"Error: Session directory {args.session_dir} does not exist.")
        return

    forward_dir = os.path.join(args.session_dir, "forward")
    down_dir = os.path.join(args.session_dir, "down")

    print("=" * 60)
    print("  YOLO Detection Video Creator")
    print("=" * 60)
    print(f"Session: {args.session_dir}")
    print("=" * 60)

    # Process forward camera
    if os.path.exists(forward_dir):
        create_video_from_dir(forward_dir)
    else:
        print(f"Note: forward directory not found in {args.session_dir}")

    # Process downward camera
    if os.path.exists(down_dir):
        create_video_from_dir(down_dir)
    else:
        print(f"Note: down directory not found in {args.session_dir}")

if __name__ == "__main__":
    main()

