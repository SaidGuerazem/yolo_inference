#!/bin/bash
# Export best.pt to a TensorRT engine on the Jetson.
# Run this on the Jetson, not on the development machine.

set -e

MODEL="${1:-best.pt}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-true}"
OUTPUT="${OUTPUT:-best.engine}"

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model file not found: $MODEL"
    exit 1
fi

if ! python3 -c "import ultralytics" >/dev/null 2>&1; then
    echo "Error: Python package 'ultralytics' is not installed on this Jetson."
    echo "Install it in the Jetson environment used for export, then rerun this script."
    exit 1
fi

echo "Exporting $MODEL to TensorRT engine..."
echo "  imgsz=$IMGSZ"
echo "  device=$DEVICE"
echo "  half=$HALF"

ultralytics export \
    model="$MODEL" \
    format=engine \
    imgsz="$IMGSZ" \
    device="$DEVICE" \
    half="$HALF"

EXPORTED="${MODEL%.*}.engine"
if [[ "$EXPORTED" != "$OUTPUT" ]]; then
    cp "$EXPORTED" "$OUTPUT"
fi

echo "Done: $OUTPUT"
