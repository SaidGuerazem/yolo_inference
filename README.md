# YOLO Detector

YOLO TensorRT detector for Jetson with ROS2 integration. This branch is for the dual MIPI/CSI camera setup only.

## Camera Setup

The detector uses two `nvarguscamerasrc` sensors composited side-by-side before inference.

- sensor `0`: left half
- sensor `1`: right half
- per-sensor capture: `1920x1200 @ 60 FPS`
- detector frame: `3840x1200 @ 20 FPS`

## Export `best.pt` To `best.engine`

Do this on the Jetson, not on the development machine:

1. Convert `best.pt` to `best.onnx` on your development PC or Jetson using Ultralytics.
2. Build the TensorRT engine from the ONNX file on the Jetson using `trtexec`:
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
   ```
   *Note: Ensure `imgsz: 1280` and `end2end: true` are used during export, which match the current configuration.*

## Build & Run (Developer Mode)

### 1. Build
Build the C++ executable:
```bash
source /opt/ros/humble/setup.bash
cd /home/sapience/ros_pkgs/src/yolo-inference-jetson/build
make -j4
```

### 2. Run
By default, the program does not save frames to disk to maximize performance (20 FPS).

* **Detection only (Fastest)**:
  ```bash
  source /opt/ros/humble/setup.bash
  ./yolo_cam_optimized --engine ../best.engine
  ```
* **Save only frames with detections**:
  ```bash
  source /opt/ros/humble/setup.bash
  ./yolo_cam_optimized --engine ../best.engine --save-detected
  ```
* **Save all processed frames**:
  ```bash
  source /opt/ros/humble/setup.bash
  ./yolo_cam_optimized --engine ../best.engine --save-all
  ```

### 3. Stop
Press `Ctrl+C` in the terminal to stop, or kill the process:
```bash
pkill -f yolo_cam_optimized
```

## Build & Install (Debian Package Mode)

Build the deb package on the Jetson:
```bash
cd /home/sapience/ros_pkgs/src/yolo-inference-jetson
./classic/build_deb.sh
sudo dpkg -i yolo-detector_1.0.0_arm64.deb
```

Once installed, you can start the detector globally from anywhere:
```bash
yolo-detector
```
or with a custom engine and options:
```bash
yolo-detector --engine /opt/yolo_detector/best.engine --save-detected
```

Press `Ctrl+C` or run `pkill -f yolo_cam_optimized` to stop.

## Image Saving

Saved frames (composited BGR from both cameras) are written to disk.
* **Save Directory**: `/home/sapience/ros_pkgs/saved_images/`
* **Annotations**: If objects are detected, green bounding boxes and confidence scores are drawn on the saved images.
* **Performance Impact**: Saving frames to disk decreases the processing rate due to disk write times (~9.5 FPS when saving every frame).
