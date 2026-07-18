#!/usr/bin/env python3
# ABOUTME: Interactive OpenCV-based labeling tool for side-angle badminton videos.
# ABOUTME: Allows frame-by-frame navigation and shot-type labeling; saves to side_labels.csv.
"""
Side-Angle Video Labeling Tool
==============================

An interactive tool to label badminton shots in side-angle training videos.

Controls:
    SPACE       Play / Pause
    D / →       Next frame
    A / ←       Previous frame
    F           Fast forward (skip 10 frames)
    B           Rewind (skip 10 frames)
    S           Mark START of shot window
    E           Mark END of shot window + prompt for shot type
    1           Label current window as SMASH
    2           Label current window as CLEAR
    3           Label current window as DROP
    Q / ESC     Quit and save

Usage:
    python -m cv_model.label_tool <video_file>
    python -m cv_model.label_tool cv_model/training_data/side_videos/smash_tutorial_insight.mp4
"""

import csv
import cv2
import os
import sys
import argparse
from datetime import datetime

LABELS_CSV = os.path.join(os.path.dirname(__file__), "training_data", "side_labels.csv")
SHOT_TYPES = {"1": "smash", "2": "clear", "3": "drop"}
COLORS = {"smash": (0, 0, 255), "clear": (255, 180, 0), "drop": (0, 200, 0)}


def load_existing_labels():
    """Load existing labels from CSV."""
    labels = []
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row)
    return labels


def save_labels(labels):
    """Save all labels to CSV."""
    fieldnames = ["video_file", "start_frame", "end_frame", "shot_type", "player_side", "labeled_at"]
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    with open(LABELS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)


def draw_info_overlay(frame, frame_idx, total_frames, fps, state, labels_count):
    """Draw the information overlay on the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent dark bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Frame info
    time_sec = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}  Time: {time_sec:.2f}s",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # State info
    if state.get("start_frame") is not None and state.get("end_frame") is None:
        cv2.putText(frame, f"MARKING: Start={state['start_frame']}  (press E to set end, 1/2/3 for type)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    elif state.get("start_frame") is not None and state.get("end_frame") is not None:
        cv2.putText(frame, f"WINDOW: {state['start_frame']}-{state['end_frame']}  Press 1=SMASH 2=CLEAR 3=DROP",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "Press S to mark shot start",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Labels saved: {labels_count}  |  Controls: SPACE=play  D/A=step  S=start  E=end  Q=quit",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    # Draw shot start marker line
    if state.get("start_frame") is not None:
        progress_start = int((state["start_frame"] / total_frames) * w)
        cv2.line(frame, (progress_start, h - 20), (progress_start, h), (0, 255, 0), 2)
    
    # Progress bar at bottom
    progress = int((frame_idx / total_frames) * w)
    cv2.rectangle(frame, (0, h - 10), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, h - 10), (progress, h), (0, 180, 255), -1)
    
    return frame


def run_labeler(video_path: str):
    """Run the interactive labeling tool."""
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    video_basename = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_basename}")
    print(f"  Resolution: {width}x{height} @ {fps:.1f}fps")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s")
    print()
    print("Controls:")
    print("  SPACE = Play/Pause")
    print("  D/→   = Next frame    A/← = Previous frame")
    print("  F     = Skip +10      B   = Skip -10")
    print("  S     = Mark start    E   = Mark end")
    print("  1     = SMASH         2   = CLEAR         3 = DROP")
    print("  Q/ESC = Quit and save")
    print()
    
    labels = load_existing_labels()
    state = {"start_frame": None, "end_frame": None}
    frame_idx = 0
    playing = False
    
    cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label Tool", min(width, 1280), min(height, 720))
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        display = draw_info_overlay(frame.copy(), frame_idx, total_frames, fps, state, len(labels))
        cv2.imshow("Label Tool", display)
        
        wait_ms = int(1000 / fps) if playing else 0
        key = cv2.waitKey(max(wait_ms, 1)) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # Space = toggle play
            playing = not playing
        elif key == ord('d') or key == 83:  # D or Right arrow
            playing = False
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == ord('a') or key == 81:  # A or Left arrow
            playing = False
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('f'):  # Fast forward
            frame_idx = min(frame_idx + 10, total_frames - 1)
        elif key == ord('b'):  # Rewind
            frame_idx = max(frame_idx - 10, 0)
        elif key == ord('s'):  # Mark start
            state["start_frame"] = frame_idx
            state["end_frame"] = None
            print(f"  ✓ Start marked at frame {frame_idx}")
        elif key == ord('e'):  # Mark end
            if state["start_frame"] is not None:
                state["end_frame"] = frame_idx
                print(f"  ✓ End marked at frame {frame_idx}. Press 1=SMASH, 2=CLEAR, 3=DROP")
        elif chr(key) in SHOT_TYPES:  # 1, 2, 3 = label shot type
            shot_type = SHOT_TYPES[chr(key)]
            if state["start_frame"] is not None:
                end = state["end_frame"] if state["end_frame"] is not None else frame_idx
                label = {
                    "video_file": video_basename,
                    "start_frame": str(state["start_frame"]),
                    "end_frame": str(end),
                    "shot_type": shot_type,
                    "player_side": "right",  # Can be adjusted
                    "labeled_at": datetime.now().isoformat(),
                }
                labels.append(label)
                print(f"  ★ Labeled: {shot_type.upper()} frames {state['start_frame']}-{end}")
                state = {"start_frame": None, "end_frame": None}
        
        if playing:
            frame_idx = min(frame_idx + 1, total_frames - 1)
            if frame_idx >= total_frames - 1:
                playing = False
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save labels
    save_labels(labels)
    print(f"\n✓ Saved {len(labels)} labels to {LABELS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label side-angle badminton videos")
    parser.add_argument("video", help="Path to the video file to label")
    args = parser.parse_args()
    run_labeler(args.video)
