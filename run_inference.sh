#!/bin/bash

# ----------------------------
# Hand-Object Detector Inference
# ----------------------------

# Video input directory
VIDEO_DIR="/Users/jing/Synapxe/semantic-segmentation/videos"

# Base output directory
BASE_SAVE_DIR="./inference-results"

# Create base output directory if it doesn't exist
mkdir -p "$BASE_SAVE_DIR"

# Find the next available numbered folder
NEXT_NUM=$(ls "$BASE_SAVE_DIR" | grep -E '^[0-9]+$' | sort -n | tail -n 1)
if [ -z "$NEXT_NUM" ]; then
    NEXT_NUM=1
else
    NEXT_NUM=$((NEXT_NUM + 1))
fi

SAVE_DIR="$BASE_SAVE_DIR/$NEXT_NUM"
mkdir -p "$SAVE_DIR"

LOADDIR=models

# Checkpoint file
CHECKSESSION=1
CHECKEPOCH=8
CHECKPOINT=89999  # just the number
# Network type
NET="res101"
FPS=30

# Debug: show which python is used and video count
echo "Using python: $(which python)"
python -V
shopt -s nullglob
VIDEOS=("$VIDEO_DIR"/*.mp4)
echo "Found ${#VIDEOS[@]} mp4 file(s) in $VIDEO_DIR"
if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "No videos found. Exiting."
    exit 1
fi

# Run inference
PYTHONUNBUFFERED=1 python -u inference.py \
    --video_dir "$VIDEO_DIR" \
    --save_dir "$SAVE_DIR" \
    --net "$NET" \
    --checksession $CHECKSESSION \
    --checkepoch $CHECKEPOCH \
    --checkpoint $CHECKPOINT \
    --load_dir $LOADDIR \
    --cuda \
    --webcam \
    --no_save
