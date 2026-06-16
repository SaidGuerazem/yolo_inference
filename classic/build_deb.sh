#!/bin/bash
# Simple .deb builder for YOLO Detector

set -e

PACKAGE_NAME="yolo-detector"
VERSION="1.0.0"
ARCH="arm64"
PACKAGE_DIR="build_deb/${PACKAGE_NAME}_${VERSION}_${ARCH}"
MODEL_ENGINE="${MODEL_ENGINE:-best.engine}"

echo "Building ${PACKAGE_NAME} v${VERSION}..."
echo "Using TensorRT engine: ${MODEL_ENGINE}"

# Setup directories
rm -rf build_deb
mkdir -p "$PACKAGE_DIR/opt/yolo_detector"
mkdir -p "$PACKAGE_DIR/usr/local/bin"
mkdir -p "$PACKAGE_DIR/DEBIAN"

# Build
make clean && make

# Copy files
cp yolo_cam_optimized "$PACKAGE_DIR/opt/yolo_detector/"
cp "$MODEL_ENGINE" "$PACKAGE_DIR/opt/yolo_detector/best.engine"
cp yolo-detector-record export_engine_jetson.sh "$PACKAGE_DIR/opt/yolo_detector/"

# Create launcher
cat > "$PACKAGE_DIR/usr/local/bin/yolo-detector" << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
cd /opt/yolo_detector
exec ./yolo_cam_optimized "$@"
EOF
chmod +x "$PACKAGE_DIR/usr/local/bin/yolo-detector"

# Create recording launcher
cat > "$PACKAGE_DIR/usr/local/bin/yolo-detector-record" << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
exec /opt/yolo_detector/yolo-detector-record "$@"
EOF
chmod +x "$PACKAGE_DIR/usr/local/bin/yolo-detector-record"

# Package metadata
cat > "$PACKAGE_DIR/DEBIAN/control" << EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Architecture: ${ARCH}
Depends: ros-humble-rclcpp, ros-humble-std-msgs
Maintainer: lsc <liam.christian@city.ac.uk>
Description: YOLO TensorRT detector with ROS2
EOF

# Post-install
cat > "$PACKAGE_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
chmod +x /opt/yolo_detector/yolo_cam_optimized /opt/yolo_detector/yolo-detector-record /opt/yolo_detector/export_engine_jetson.sh
chmod +x /usr/local/bin/yolo-detector /usr/local/bin/yolo-detector-record
echo "Installed! Run with:"
echo "  yolo-detector              (detection only)"
echo "  yolo-detector-record       (detection + raw video)"
echo "  /opt/yolo_detector/export_engine_jetson.sh"
EOF
chmod +x "$PACKAGE_DIR/DEBIAN/postinst"

# Build package
dpkg-deb --build "$PACKAGE_DIR"
mv "build_deb/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb" .
rm -rf build_deb

echo ""
echo "Done! Package: ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo "Install: sudo dpkg -i ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo "Run: yolo-detector"
