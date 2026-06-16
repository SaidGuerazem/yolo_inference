import os
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO format labels on saved raw images.")
    parser.add_argument("dir", type=str, help="Directory containing JPEGs and text files (e.g. forward or down)")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist.")
        return

    output_dir = os.path.join(args.dir, "annotated")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.dir) if f.endswith(".jpg")]
    if not files:
        print("No JPEGs found in the directory.")
        return

    print(f"Processing {len(files)} JPEGs in {args.dir}...")
    annotated_count = 0

    for filename in files:
        img_path = os.path.join(args.dir, filename)
        txt_path = os.path.join(args.dir, filename.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {filename}")
            continue

        h, w, _ = img.shape

        if os.path.exists(txt_path):
            annotated_count += 1
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

                xmin = int(cx - box_w / 2.0)
                ymin = int(cy - box_h / 2.0)
                xmax = int(cx + box_w / 2.0)
                ymax = int(cy + box_h / 2.0)

                # Draw bounding box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Draw label
                label = f"Class {class_id}"
                cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, img)

    print(f"Done! Created {annotated_count} annotated images under {output_dir}")

if __name__ == "__main__":
    main()

