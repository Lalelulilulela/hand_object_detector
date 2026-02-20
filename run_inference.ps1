#!/usr/bin/env pwsh

# ----------------------------
# Hand-Object Detector Inference
# ----------------------------
# Video input directory
$VIDEO_DIR = "C:\Users\DSAI_Team\Desktop\SPPB\SPPB_balance_model4\dataset\videos\object_contact_test"

# Base output directory
$BASE_SAVE_DIR = "./inference-results-object_contact_test-132028"

# Create base output directory if it doesn't exist
New-Item -ItemType Directory -Path $BASE_SAVE_DIR -Force | Out-Null

# Find the next available numbered folder
$existingNumbers = @()
if (Test-Path $BASE_SAVE_DIR) {
    $existingNumbers = Get-ChildItem -Path $BASE_SAVE_DIR -Directory |
        Where-Object { $_.Name -match '^[0-9]+$' } |
        ForEach-Object { [int]$_.Name } |
        Sort-Object
}

if ($existingNumbers.Count -eq 0) {
    $NEXT_NUM = 1
} else {
    $NEXT_NUM = $existingNumbers[-1] + 1
}

$SAVE_DIR = Join-Path (Resolve-Path $BASE_SAVE_DIR) $NEXT_NUM
New-Item -ItemType Directory -Path $SAVE_DIR -Force | Out-Null

$LOADDIR = "models"

# Checkpoint file
$CHECKSESSION = 1
$CHECKEPOCH = 8
$CHECKPOINT = 132028  # just the number
# Network type
$NET = "res101"
$FPS = 30

# Debug: show which python is used and video count
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "python not found in PATH. Exiting."
    exit 1
}

Write-Host ("Using python: {0}" -f $pythonCmd.Source)
python -V

$VIDEOS = Get-ChildItem -Path $VIDEO_DIR -Filter "*.mp4" -File -ErrorAction SilentlyContinue
Write-Host ("Found {0} mp4 file(s) in {1}" -f $VIDEOS.Count, $VIDEO_DIR)
if ($VIDEOS.Count -eq 0) {
    Write-Host "No videos found. Exiting."
    exit 1
}

# Run inference
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUNBUFFERED = "1"

python -s -u inference.py `
    --video_dir "$VIDEO_DIR" `
    --save_dir "$SAVE_DIR" `
    --net "$NET" `
    --checksession $CHECKSESSION `
    --checkepoch $CHECKEPOCH `
    --checkpoint $CHECKPOINT `
    --load_dir $LOADDIR `
    --cuda
    # --webcam `
    # --no_save
