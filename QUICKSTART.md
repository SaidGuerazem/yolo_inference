# YOLO Detector - Quick Start

## Build & Run

To start the detector from the source folder:

```bash
# 1. Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# 2. Navigate to the build folder and compile
cd /home/sapience/ros_pkgs/src/yolo-inference-jetson/build
make -j4

# 3. Run the detector
./yolo_cam_optimized --engine ../best.engine
```

To stop the detector, press **`Ctrl+C`** or run:
```bash
pkill -f yolo_cam_optimized
```

## Running with Image Saving Options

By default, no images are saved to disk to preserve the full processing speed (20 FPS). You can enable frame saving using:

* **Save only frames where objects are detected**:
  ```bash
  ./yolo_cam_optimized --engine ../best.engine --save-detected
  ```
* **Save every single frame**:
  ```bash
  ./yolo_cam_optimized --engine ../best.engine --save-all
  ```

Saved images are placed in `/home/sapience/ros_pkgs/saved_images/`.

## Running as a ROS 2 Package

If you build the workspace using `colcon`:
```bash
cd /home/sapience/ros_pkgs
colcon build --packages-select yolo-inference
source install/setup.bash
ros2 run yolo-inference yolo_cam_optimized --engine src/yolo-inference-jetson/best.engine --save-detected
```

## Camera Defaults

The detector is MIPI/CSI-only.

| Setting | Default |
|---------|---------|
| Sensor IDs | `0` left, `1` right |
| Per-sensor capture | `1920x1200 @ 60 FPS` |
| Composited detector frame | `3840x1200 @ 20 FPS` |

Override values only if the camera mapping changes:
```bash
./yolo_cam_optimized --sensor0 0 --sensor1 1 --width 3840 --height 1200 --fps 20
```
