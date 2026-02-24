#!/bin/bash

LOADDIR=models
CHECKSESSION=1
CHECKEPOCH=8
NET="res101"
CHECKPOINT=132028
FPS=30
THRESH_HAND=0.6
THRESH_OBJ=0.7
THRESH_CONTACT=0.5
HAND_STATES="0"
VIDEO_DIR="/Users/jing/Synapxe/semantic-segmentation/videos"  # Video input directory
BASE_SAVE_DIR="./inference-results-no_contact_test-$CHECKPOINT"    # Base output directory

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
ARGS=(
    --video_dir "$VIDEO_DIR"
    --save_dir "$SAVE_DIR"
    --net "$NET"
    --checksession "$CHECKSESSION"
    --checkepoch "$CHECKEPOCH"
    --checkpoint "$CHECKPOINT"
    --load_dir "$LOADDIR"
    --thresh_hand "$THRESH_HAND"
    --thresh_obj "$THRESH_OBJ"
    --thresh_contact "$THRESH_CONTACT"
    --hand_states "$HAND_STATES"
    # --webcam
    # --no_save
)

PYTHONUNBUFFERED=1 python -u inference.py "${ARGS[@]}"
