#!/usr/bin/env python3
# ABOUTME: Production training pipeline with fine-tuning, experiment tracking, and multi-dataset support.
# ABOUTME: Supports train-from-scratch, fine-tuning from pretrained weights, and resume-training workflows.
"""
Training Pipeline for Badminton Shot Classifier
================================================

Supports:
  - Training from scratch (random init)
  - Fine-tuning from pretrained weights (--pretrained)
  - Resuming interrupted training (--resume)
  - Multiple dataset formats:
      * ShuttleSet CSV annotations (broadcast mode)
      * Side-angle labels from label_tool.py (side mode)
      * VideoBadminton folder-based dataset (--dataset-dir)
  - Advanced biomechanically-safe data augmentation
  - Experiment tracking with auto-versioned runs/
  - Comprehensive evaluation with plots and metrics

Usage:
    # Train from scratch on VideoBadminton dataset
    python -m cv_model.train_model --mode broadcast --dataset-dir path/to/VideoBadminton_Dataset

    # Fine-tune from existing checkpoint
    python -m cv_model.train_model --mode side --pretrained cv_model/models/shot_classifier_side_v1.keras --lr 5e-5

    # Resume interrupted training
    python -m cv_model.train_model --resume cv_model/runs/experiment_001/last_model.keras
"""

import csv
import os
import sys
import glob
import json
import logging
import argparse
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

SHUTTLESET_DIR = os.path.join(os.path.dirname(__file__), "..", "temp_coachAI", "ShuttleSet", "set")
MATCH_CSV = os.path.join(SHUTTLESET_DIR, "match.csv")

# Map Chinese shot type names to English categories
SHOT_TYPE_MAP = {
    # Smash variants
    "殺球": "smash",
    "點扣": "smash",       # dab smash is still a smash
    # Clear
    "長球": "clear",
    # Drop variants  
    "切球": "drop",
    "過度切球": "drop",    # transition drop
    # Lob
    "挑球": "lob",
    # Drive variants
    "平球": "drive",
    "後場抽平球": "drive",  # backcourt drive
    "防守回抽": "drive",    # defensive drive
    # Net shots
    "放小球": "net",
    "勾球": "net",         # cross-net
    "擋小球": "net",       # block at net
    # Push/rush
    "推球": "push",
    "撲球": "rush",
    # Serves (skip for shot classification)
    "發短球": "serve",
    "發長球": "serve",
    # Defensive lob
    "防守回挑": "lob",
}

# The 3 shot classes we care about for our model
TARGET_CLASSES = ["clear", "drop", "smash"]
CLASS_TO_ID = {c: i for i, c in enumerate(TARGET_CLASSES)}

# Training constants
WINDOW_SIZE = 41  # frames per shot sequence
NUM_KEYPOINTS = 13
NUM_COORDS = 2  # x, y
FEATURE_DIM = NUM_KEYPOINTS * NUM_COORDS + 8  # 26 + 8 engineered = 34
SIDE_FEATURE_DIM = NUM_KEYPOINTS * NUM_COORDS + 16  # 26 + 16 engineered = 42

# Side-angle labels file
SIDE_LABELS_CSV = os.path.join(os.path.dirname(__file__), "training_data", "side_labels.csv")
SIDE_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "training_data", "side_videos")

# VideoBadminton dataset: folder name → our class mapping
VIDEOBADMINTON_CLASS_MAP = {
    "03_Tap Smash": "smash",
    "14_Smash": "smash",
    "05_Drop Shot": "drop",
    "08_Cut": "drop",         # cut is a drop variant
    "12_Clear": "clear",
    "10_Defensive Clear": "clear",
}


# ─────────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Centralized hyperparameter configuration with CLI override support."""
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    dropout: float = 0.3
    window_size: int = WINDOW_SIZE
    window_stride: int = 10
    lstm_units: int = 128
    conv_filters_1: int = 64
    conv_filters_2: int = 128
    optimizer: str = "adam"          # "adam" or "adamw"
    weight_decay: float = 1e-4
    augment: bool = True
    seed: int = 42
    patience_early_stop: int = 20
    patience_reduce_lr: int = 7
    min_lr: float = 1e-7
    reduce_lr_factor: float = 0.5
    max_per_class: int = 500
    mode: str = "broadcast"         # "broadcast" or "side"
    pretrained: str = ""
    resume: str = ""

    def to_dict(self):
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ShotSample:
    """A single labeled shot sample."""
    match_name: str
    set_num: int
    rally: int
    ball_round: int
    frame_num: int
    shot_type: str  # English name
    class_id: int


# ─────────────────────────────────────────────────────────────────────
# Dataset Loaders
# ─────────────────────────────────────────────────────────────────────

def load_shuttleset_annotations(max_per_class: int = 500, video_dir: str = None) -> List[ShotSample]:
    """
    Parse all ShuttleSet CSV files and extract shot annotations.
    
    Only loads samples from matches that have a downloaded video in video_dir.
    
    Returns a list of ShotSample objects for the target classes (clear, drop, smash).
    """
    samples = []
    class_counts = {c: 0 for c in TARGET_CLASSES}

    # Read match.csv to get video-to-folder mapping
    match_info = {}
    with open(MATCH_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            match_info[row["video"]] = row

    # Get list of downloaded videos (by basename without extension)
    available_videos = set()
    if video_dir and os.path.isdir(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                available_videos.add(os.path.splitext(f)[0])
    logger.info(f"Available downloaded videos: {len(available_videos)}")

    # Parse all set CSVs
    csv_files = glob.glob(os.path.join(SHUTTLESET_DIR, "*", "set*.csv"))
    logger.info(f"Found {len(csv_files)} set CSV files")

    for csv_file in csv_files:
        match_dir = os.path.basename(os.path.dirname(csv_file))

        # Skip if this match has no downloaded video
        if available_videos and match_dir not in available_videos:
            continue
        set_name = os.path.splitext(os.path.basename(csv_file))[0]
        set_num = int(set_name.replace("set", ""))

        with open(csv_file, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                shot_type_cn = row.get("type", "")
                shot_type_en = SHOT_TYPE_MAP.get(shot_type_cn)

                if shot_type_en not in TARGET_CLASSES:
                    continue

                # Check if we have enough of this class
                if class_counts[shot_type_en] >= max_per_class:
                    continue

                frame_num_str = row.get("frame_num", "")
                if not frame_num_str or frame_num_str == "":
                    continue

                try:
                    frame_num = int(float(frame_num_str))
                except (ValueError, TypeError):
                    continue

                sample = ShotSample(
                    match_name=match_dir,
                    set_num=set_num,
                    rally=int(row.get("rally", 0)),
                    ball_round=int(float(row.get("ball_round", 0))),
                    frame_num=frame_num,
                    shot_type=shot_type_en,
                    class_id=CLASS_TO_ID[shot_type_en],
                )
                samples.append(sample)
                class_counts[shot_type_en] += 1

    logger.info(f"Loaded {len(samples)} shot annotations:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count}")

    return samples


def load_side_angle_annotations(video_dir: str = None) -> List[ShotSample]:
    """
    Load shot annotations from side_labels.csv (created by label_tool.py).
    
    Returns a list of ShotSample objects for the target classes.
    """
    if not os.path.exists(SIDE_LABELS_CSV):
        logger.error(f"Side-angle labels not found: {SIDE_LABELS_CSV}")
        logger.error("Run the labeling tool first: python -m cv_model.label_tool <video>")
        return []

    samples = []
    class_counts = {c: 0 for c in TARGET_CLASSES}

    with open(SIDE_LABELS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shot_type = row.get("shot_type", "")
            if shot_type not in TARGET_CLASSES:
                continue

            video_file = row.get("video_file", "")
            start_frame = int(row.get("start_frame", 0))
            end_frame = int(row.get("end_frame", 0))

            # Check video exists
            if video_dir:
                video_path = os.path.join(video_dir, video_file)
                if not os.path.exists(video_path):
                    continue

            # Use midpoint of the labeled window as the "frame_num"
            mid_frame = (start_frame + end_frame) // 2

            sample = ShotSample(
                match_name=video_file,
                set_num=0,
                rally=0,
                ball_round=0,
                frame_num=mid_frame,
                shot_type=shot_type,
                class_id=CLASS_TO_ID[shot_type],
            )
            samples.append(sample)
            class_counts[shot_type] += 1

    logger.info(f"Loaded {len(samples)} side-angle annotations:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count}")

    return samples


def load_videobadminton_clips(
    dataset_dir: str,
    class_map: Dict[str, str] = None,
    max_per_class: int = 500,
    seed: int = 42,
) -> List[Tuple[str, str, int]]:
    """
    Load video clips from VideoBadminton folder-based dataset.
    
    Each subfolder = one shot type. Each .mp4 = one short clip of a single shot.
    
    Args:
        dataset_dir: Root of VideoBadminton_Dataset (contains 00_Short Serve, etc.)
        class_map: Mapping of folder names to our target classes.
        max_per_class: Maximum clips per class to load.
        seed: Random seed for shuffling clips.
        
    Returns:
        List of (video_path, class_name, class_id) tuples.
    """
    if class_map is None:
        class_map = VIDEOBADMINTON_CLASS_MAP

    rng = np.random.RandomState(seed)
    clips = []
    class_counts = {c: 0 for c in TARGET_CLASSES}

    for folder_name, class_name in class_map.items():
        if class_name not in TARGET_CLASSES:
            continue
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            logger.warning(f"  Folder not found: {folder_path}")
            continue

        mp4s = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))
        rng.shuffle(mp4s)  # Shuffle to avoid temporal bias when capping

        for mp4 in mp4s:
            if class_counts[class_name] >= max_per_class:
                break
            clips.append((mp4, class_name, CLASS_TO_ID[class_name]))
            class_counts[class_name] += 1

    logger.info(f"Loaded {len(clips)} VideoBadminton clips:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count}")

    return clips


# ─────────────────────────────────────────────────────────────────────
# Pose Sequence Extraction
# ─────────────────────────────────────────────────────────────────────

def extract_videobadminton_sequences(
    clips: List[Tuple[str, str, int]],
    window_size: int = WINDOW_SIZE,
    feature_mode: str = "broadcast",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pose sequences from VideoBadminton short video clips.
    
    Each clip IS one shot (~29 frames). We center-pad shorter clips
    to the full window_size and extract features.
    
    Args:
        clips: List of (video_path, class_name, class_id) from load_videobadminton_clips.
        window_size: Target sequence length (default 41).
        feature_mode: "broadcast" (34-dim) or "side" (42-dim).
        
    Returns:
        X: (N, window_size, feature_dim)
        y: (N,)
    """
    from cv_model.pose_extractor import PoseExtractor
    import cv2

    extractor = PoseExtractor()
    X_list = []
    y_list = []
    skipped = 0

    for i, (video_path, class_name, class_id) in enumerate(clips):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            skipped += 1
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_dim = float(max(frame_w, frame_h))

        # Raw keypoint sequence
        sequence = np.zeros((window_size, NUM_KEYPOINTS, NUM_COORDS), dtype=np.float32)

        # Center the clip frames within the window
        clip_len = min(total_frames, window_size)
        offset = max(0, (window_size - clip_len) // 2)

        for f_idx in range(clip_len):
            ret, frame = cap.read()
            if not ret:
                break

            poses = extractor.extract_single_frame(frame, normalize=False)
            if poses:
                # Pick person with largest bounding box
                best_pose = max(
                    poses,
                    key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]),
                )
                kps = best_pose.keypoints_xy / max_dim
                sequence[offset + f_idx] = kps

        cap.release()

        # Validate: need at least 30% non-zero frames (lower for short clips)
        non_zero = np.sum(np.any(sequence != 0, axis=(1, 2)))
        if non_zero < window_size * 0.3:
            skipped += 1
            continue

        # Compute features
        if feature_mode == "side":
            from cv_model.features import compute_side_angle_features
            enhanced = compute_side_angle_features(sequence)
        else:
            from cv_model.features import compute_kinematic_features
            enhanced = compute_kinematic_features(sequence)

        X_list.append(enhanced)
        y_list.append(class_id)

        if (i + 1) % 100 == 0:
            logger.info(f"  Extracted {i + 1}/{len(clips)} clips ({skipped} skipped)...")

    if not X_list:
        logger.error("No valid sequences extracted from VideoBadminton clips!")
        return np.array([]), np.array([])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Extracted {len(X)} sequences (skipped {skipped}): X shape={X.shape}")
    for cls_name, cls_id in CLASS_TO_ID.items():
        count = np.sum(y == cls_id)
        logger.info(f"  {cls_name}: {count}")

    return X, y


def extract_side_pose_sequences(
    samples: List[ShotSample],
    video_dir: str,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[ShotSample]]:
    """
    Extract pose sequences from side-angle videos for each shot sample.
    Uses compute_side_angle_features for expanded biomechanical features.
    
    Returns:
        X: numpy array of shape (N, window_size, 42)
        y: numpy array of shape (N,) with class IDs
        valid_samples: list of samples that were successfully extracted
    """
    from cv_model.pose_extractor import PoseExtractor

    # Group samples by video file
    samples_by_video = {}
    for s in samples:
        video_path = os.path.join(video_dir, s.match_name)
        if not os.path.exists(video_path):
            continue
        if video_path not in samples_by_video:
            samples_by_video[video_path] = []
        samples_by_video[video_path].append(s)

    logger.info(f"Found {len(samples_by_video)} side-angle videos with matching samples")
    total_samples = sum(len(v) for v in samples_by_video.values())
    logger.info(f"Total extractable samples: {total_samples}")

    X_list = []
    y_list = []
    valid_samples = []
    extractor = PoseExtractor()
    half_window = window_size // 2

    for video_path, video_samples in samples_by_video.items():
        logger.info(f"Processing: {os.path.basename(video_path)} ({len(video_samples)} shots)")

        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_dim = float(max(frame_w, frame_h))
        logger.info(f"  Video: {frame_w}x{frame_h} @ {fps}fps, {total_frames} frames")

        for sample in video_samples:
            start_frame = max(0, sample.frame_num - half_window)
            end_frame = min(total_frames - 1, sample.frame_num + half_window)

            sequence = np.zeros((window_size, NUM_KEYPOINTS, NUM_COORDS), dtype=np.float32)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for i in range(window_size):
                frame_idx = start_frame + i
                if frame_idx > end_frame:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                poses = extractor.extract_single_frame(frame, normalize=False)
                if poses:
                    # In side-angle videos, pick the largest detected person
                    best_pose = max(poses, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
                    kps = best_pose.keypoints_xy  # (13, 2) in pixels
                    kps_normalized = kps / max_dim
                    sequence[i] = kps_normalized

            # Check valid frames (at least 50% non-zero)
            non_zero_frames = np.sum(np.any(sequence != 0, axis=(1, 2)))
            if non_zero_frames < window_size * 0.5:
                continue

            # Apply SIDE-ANGLE feature engineering: (41, 13, 2) -> (41, 42)
            from cv_model.features import compute_side_angle_features
            enhanced_seq = compute_side_angle_features(sequence)

            X_list.append(enhanced_seq)
            y_list.append(sample.class_id)
            valid_samples.append(sample)

        cap.release()

    if not X_list:
        logger.error("No valid pose sequences extracted!")
        return np.array([]), np.array([]), []

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Extracted {len(X)} pose sequences: X shape={X.shape}, y shape={y.shape}")
    for cls_name, cls_id in CLASS_TO_ID.items():
        count = np.sum(y == cls_id)
        logger.info(f"  {cls_name}: {count}")

    return X, y, valid_samples


def extract_pose_sequences(
    samples: List[ShotSample],
    video_dir: str,
    match_csv_path: str,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[ShotSample]]:
    """
    Extract pose sequences from videos for each shot sample.
    
    For each shot at frame F, extracts frames [F - window_size//2, F + window_size//2].
    
    Returns:
        X: numpy array of shape (N, window_size, 26)
        y: numpy array of shape (N,) with class IDs
        valid_samples: list of samples that were successfully extracted
    """
    from cv_model.pose_extractor import PoseExtractor

    # Load match info for video name -> YouTube URL mapping
    match_to_video = {}
    with open(match_csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            match_to_video[row["video"]] = row

    # Group samples by match name to process each video only once
    samples_by_match = {}
    for s in samples:
        # Find the video name for this match directory
        video_name = None
        for vid_name, info in match_to_video.items():
            if vid_name == s.match_name or s.match_name.startswith(vid_name):
                video_name = vid_name
                break
        if video_name is None:
            # Try matching by removing spaces/underscores
            for vid_name, info in match_to_video.items():
                if vid_name.replace(" ", "_").replace("__", "_") == s.match_name.replace(" ", "_").replace("__", "_"):
                    video_name = vid_name
                    break
        
        if video_name is None:
            continue

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            continue

        if video_path not in samples_by_match:
            samples_by_match[video_path] = []
        samples_by_match[video_path].append(s)

    logger.info(f"Found {len(samples_by_match)} videos with matching samples")
    total_samples = sum(len(v) for v in samples_by_match.values())
    logger.info(f"Total extractable samples: {total_samples}")

    X_list = []
    y_list = []
    valid_samples = []
    extractor = PoseExtractor()
    half_window = window_size // 2

    for video_path, video_samples in samples_by_match.items():
        logger.info(f"Processing: {os.path.basename(video_path)} ({len(video_samples)} shots)")

        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_dim = float(max(frame_w, frame_h))
        logger.info(f"  Video: {frame_w}x{frame_h} @ {fps}fps, {total_frames} frames")

        for sample in video_samples:
            start_frame = max(0, sample.frame_num - half_window)
            end_frame = min(total_frames - 1, sample.frame_num + half_window)

            # Extract frames
            sequence = np.zeros((window_size, NUM_KEYPOINTS, NUM_COORDS), dtype=np.float32)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for i in range(window_size):
                frame_idx = start_frame + i
                if frame_idx > end_frame:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Extract pose (unnormalized — we normalize ourselves)
                poses = extractor.extract_single_frame(frame, normalize=False)
                if poses:
                    # Pick the person with the largest bounding box (likely the player)
                    best_pose = max(poses, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
                    kps = best_pose.keypoints_xy  # (13, 2) in pixels

                    # Normalize by max dimension to preserve aspect ratio
                    # This is our CANONICAL normalization — shared between train and inference
                    kps_normalized = kps / max_dim
                    sequence[i] = kps_normalized

            # Check that we got enough valid frames (at least 50% non-zero)
            non_zero_frames = np.sum(np.any(sequence != 0, axis=(1, 2)))
            if non_zero_frames < window_size * 0.5:
                continue

            # Apply kinematic physics feature engineering: (41, 13, 2) -> (41, 34)
            from cv_model.features import compute_kinematic_features
            enhanced_seq = compute_kinematic_features(sequence)
            
            X_list.append(enhanced_seq)
            y_list.append(sample.class_id)
            valid_samples.append(sample)

        cap.release()

    if not X_list:
        logger.error("No valid pose sequences extracted!")
        return np.array([]), np.array([]), []

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Extracted {len(X)} pose sequences: X shape={X.shape}, y shape={y.shape}")
    for cls_name, cls_id in CLASS_TO_ID.items():
        count = np.sum(y == cls_id)
        logger.info(f"  {cls_name}: {count}")

    return X, y, valid_samples


# ─────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────

def build_advanced_model(input_shape: Tuple[int, int], num_classes: int, config: TrainingConfig = None):
    """Build the Conv1D + Bidirectional LSTM + Attention model."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Bidirectional, LSTM, Dense, Dropout, Multiply

    if config is None:
        config = TrainingConfig()

    dropout = config.dropout

    inputs = Input(shape=input_shape)

    # 1. Spatial-Temporal Convolutions
    # Sliding window over the frames to extract local velocity/acceleration features
    x = Conv1D(filters=config.conv_filters_1, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=config.conv_filters_2, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 2. Temporal Memory (Bidirectional LSTM)
    x = Bidirectional(LSTM(config.lstm_units, return_sequences=True))(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(config.lstm_units, return_sequences=False))(x)
    x = Dropout(dropout)(x)

    # 3. Squeeze-and-Excitation (Attention Mechanism)
    # Helps the model focus on the most important temporal features
    se = Dense(64, activation='relu')(x)
    se = Dense(config.lstm_units * 2, activation='sigmoid')(se)
    x = Multiply()([x, se])

    # 4. Classifier
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile with optimizer from config
    if config.optimizer == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────────

def augment_data(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic data augmentation (legacy, preserved for backward compatibility).
    Applies Jittering and Mirroring. Triples dataset size.
    """
    logger.info("Applying Basic Data Augmentation (Jittering + Mirroring)...")
    X_aug = [X_train]
    y_aug = [y_train]

    # 1. Jittering (Add small Gaussian noise)
    noise = np.random.normal(loc=0, scale=0.01, size=X_train.shape)
    X_jitter = X_train + noise
    X_jitter = np.clip(X_jitter, 0.0, 1.0)
    X_aug.append(X_jitter)
    y_aug.append(y_train)

    # 2. Mirroring (Flip left/right to simulate both handedness)
    X_mirror = X_train.copy()
    X_mirror[:, :, 0::2] = 1.0 - X_mirror[:, :, 0::2]
    X_aug.append(X_mirror)
    y_aug.append(y_train)

    X_final = np.concatenate(X_aug, axis=0)
    y_final = np.concatenate(y_aug, axis=0)
    logger.info(f"Augmented data size: {len(X_train)} -> {len(X_final)}")
    return X_final, y_final


def augment_data_advanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced biomechanically-safe data augmentation for pose sequences.
    
    Augmentations applied:
      1. Gaussian noise (σ=0.005) — simulates pose estimation jitter
      2. Magnitude scaling (0.9–1.1×) — simulates different body sizes / camera distances
      3. Temporal shift (±3 frames) — simulates different labeling precision
      4. Speed variation (0.8–1.2×) — simulates different swing tempos
      5. Random frame drop (2–4 frames) — simulates detection failures
    
    All augmentations are safe to apply at the feature level.
    
    Args:
        X_train: (N, T, F) feature sequences
        y_train: (N,) class labels
        seed: random seed
        
    Returns:
        Augmented (X, y) — approximately 6× original size.
    """
    logger.info("Applying Advanced Data Augmentation...")
    rng = np.random.RandomState(seed)
    N, T, F = X_train.shape

    X_aug = [X_train.copy()]
    y_aug = [y_train.copy()]

    # ── 1. Gaussian Noise (σ=0.005) ──
    noise = rng.normal(0, 0.005, X_train.shape).astype(np.float32)
    X_noisy = np.clip(X_train + noise, -1.0, 2.0)  # Allow small excursions
    X_aug.append(X_noisy)
    y_aug.append(y_train)

    # ── 2. Magnitude Scaling (0.9–1.1×) ──
    scales = rng.uniform(0.9, 1.1, size=(N, 1, 1)).astype(np.float32)
    X_scaled = X_train * scales
    X_aug.append(X_scaled)
    y_aug.append(y_train)

    # ── 3. Temporal Shift (±3 frames) ──
    X_shifted = np.zeros_like(X_train)
    for i in range(N):
        shift = rng.randint(-3, 4)  # -3 to +3
        if shift > 0:
            X_shifted[i, shift:] = X_train[i, :-shift]
            X_shifted[i, :shift] = X_train[i, 0]  # Edge-pad with first frame
        elif shift < 0:
            X_shifted[i, :shift] = X_train[i, -shift:]
            X_shifted[i, shift:] = X_train[i, -1]  # Edge-pad with last frame
        else:
            X_shifted[i] = X_train[i]
    X_aug.append(X_shifted)
    y_aug.append(y_train)

    # ── 4. Speed Variation (0.8–1.2× via linear interpolation) ──
    X_speed = np.zeros_like(X_train)
    for i in range(N):
        speed = rng.uniform(0.8, 1.2)
        old_indices = np.arange(T)
        # New time indices (stretched or compressed)
        new_len = int(T * speed)
        new_indices = np.linspace(0, T - 1, new_len)
        # Interpolate each feature channel
        resampled = np.zeros((new_len, F), dtype=np.float32)
        for f_idx in range(F):
            resampled[:, f_idx] = np.interp(new_indices, old_indices, X_train[i, :, f_idx])
        # Center-crop or center-pad back to T frames
        if new_len >= T:
            start = (new_len - T) // 2
            X_speed[i] = resampled[start : start + T]
        else:
            start = (T - new_len) // 2
            X_speed[i, start : start + new_len] = resampled
    X_aug.append(X_speed)
    y_aug.append(y_train)

    # ── 5. Random Frame Drop (drop 2–4 frames, linearly interpolate) ──
    X_dropped = X_train.copy()
    for i in range(N):
        n_drop = rng.randint(2, 5)  # Drop 2, 3, or 4 frames
        drop_indices = sorted(rng.choice(range(1, T - 1), size=min(n_drop, T - 2), replace=False))
        for d_idx in drop_indices:
            # Find nearest non-dropped neighbors
            prev_idx = d_idx - 1
            next_idx = d_idx + 1
            while next_idx in drop_indices and next_idx < T - 1:
                next_idx += 1
            # Linear interpolation
            alpha = 0.5
            X_dropped[i, d_idx] = (1 - alpha) * X_train[i, prev_idx] + alpha * X_train[i, next_idx]
    X_aug.append(X_dropped)
    y_aug.append(y_train)

    # ── Combine and shuffle ──
    X_final = np.concatenate(X_aug, axis=0)
    y_final = np.concatenate(y_aug, axis=0)

    perm = rng.permutation(len(X_final))
    X_final = X_final[perm]
    y_final = y_final[perm]

    logger.info(f"Advanced augmentation: {N} → {len(X_final)} samples (6× expansion)")
    return X_final, y_final


# ─────────────────────────────────────────────────────────────────────
# Evaluation & Visualization
# ─────────────────────────────────────────────────────────────────────

def generate_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history,
    output_dir: str,
    target_classes: List[str],
) -> dict:
    """
    Generate comprehensive evaluation artifacts after training.
    
    Produces:
      - Classification report (logged)
      - confusion_matrix.png
      - training_curves.png
      - metrics.json
      
    Returns:
        Dictionary of metrics.
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    num_classes = len(target_classes)

    # ── Predictions ──
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)

    # ── Classification report ──
    report_str = classification_report(y_true, y_pred, target_names=target_classes, zero_division=0)
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    logger.info(f"\n{report_str}")

    # ── Test loss & accuracy ──
    y_test_cat = to_categorical(y_true, num_classes) if y_test.ndim == 1 else y_test
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # ── Per-class metrics ──
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )

    # ── Build metrics dict ──
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "per_class": {},
        "macro_avg": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1": float(np.mean(f1)),
        },
        "confusion_matrix": cm.tolist(),
    }
    for i, cls in enumerate(target_classes):
        metrics["per_class"][cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # ── Save metrics JSON ──
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # ── Generate plots (graceful degradation if matplotlib unavailable) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_classes, fontsize=11)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_classes, fontsize=11)
        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black",
                )
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_path}")

        # Training curves
        if history and hasattr(history, "history"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
            ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
            ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            ax2.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
            ax2.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
            ax2.set_title("Accuracy Curves", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            curves_path = os.path.join(output_dir, "training_curves.png")
            plt.savefig(curves_path, dpi=150)
            plt.close()
            logger.info(f"Training curves saved to: {curves_path}")

    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation.")
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    return metrics


# ─────────────────────────────────────────────────────────────────────
# Experiment Management
# ─────────────────────────────────────────────────────────────────────

def get_next_experiment_dir(base_dir: str = None) -> str:
    """Create next experiment directory with auto-incrementing number."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir) if d.startswith("experiment_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if existing:
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_")[1]))
            except (ValueError, IndexError):
                pass
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    exp_dir = os.path.join(base_dir, f"experiment_{next_num:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# ─────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig = None,
    pretrained: str = None,
    resume: str = None,
    experiment_dir: str = None,
    output_dir: str = "cv_model/models",
    mode: str = "broadcast",
):
    """
    Train, fine-tune, or resume-train the BiLSTM shot classifier.
    
    Args:
        X: Feature sequences, shape (N, T, F).
        y: Class labels, shape (N,).
        config: TrainingConfig with all hyperparameters.
        pretrained: Path to pretrained .keras to fine-tune from.
        resume: Path to checkpoint to resume training from.
        experiment_dir: Where to save this experiment's artifacts.
        output_dir: Legacy model output directory.
        mode: "broadcast" or "side" — controls legacy model filename.
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        CSVLogger,
    )
    from sklearn.model_selection import train_test_split

    if config is None:
        config = TrainingConfig()

    if experiment_dir is None:
        experiment_dir = get_next_experiment_dir()

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num_classes = len(TARGET_CLASSES)

    # ── Reproducibility ──
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # ── Stratified train/val/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=config.seed, stratify=y_train
    )

    logger.info(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Data Augmentation (training set only) ──
    if config.augment:
        X_train, y_train = augment_data_advanced(X_train, y_train, seed=config.seed)
    else:
        X_train, y_train = augment_data(X_train, y_train)

    logger.info(f"After augmentation — Train: {len(X_train)}")

    # ── One-hot encode labels ──
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # ── Build or Load Model ──
    input_shape = (X_train.shape[1], X_train.shape[2])

    if resume:
        logger.info(f"═══ RESUMING training from: {resume}")
        from tensorflow.keras.layers import LSTM as LSTM_Layer

        class LegacyLSTM(LSTM_Layer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        try:
            model = tf.keras.models.load_model(resume)
        except TypeError:
            model = tf.keras.models.load_model(resume, custom_objects={"LSTM": LegacyLSTM})
        logger.info("  Resumed model loaded. Optimizer state preserved.")

    elif pretrained:
        logger.info(f"═══ FINE-TUNING from pretrained: {pretrained}")
        from tensorflow.keras.layers import LSTM as LSTM_Layer

        class LegacyLSTM(LSTM_Layer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        try:
            model = tf.keras.models.load_model(pretrained, compile=False)
        except TypeError:
            model = tf.keras.models.load_model(
                pretrained, custom_objects={"LSTM": LegacyLSTM}, compile=False
            )

        # Recompile with fine-tuning optimizer (lower LR)
        if config.optimizer == "adamw":
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=config.lr, weight_decay=config.weight_decay
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        logger.info(f"  Pretrained weights loaded. Recompiled with lr={config.lr}, optimizer={config.optimizer}")

    else:
        logger.info("═══ TRAINING FROM SCRATCH")
        model = build_advanced_model(input_shape, num_classes, config)

    model.summary()

    # ── Callbacks ──
    best_model_path = os.path.join(experiment_dir, "best_model.keras")
    last_model_path = os.path.join(experiment_dir, "last_model.keras")
    history_csv_path = os.path.join(experiment_dir, "history.csv")

    # Determine legacy model name based on mode
    if mode == "side":
        legacy_name = "shot_classifier_side_v1.keras"
    else:
        legacy_name = "shot_classifier_v2.keras"
    legacy_model_path = os.path.join(output_dir, legacy_name)

    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
        ModelCheckpoint(last_model_path, save_best_only=False, verbose=0),
        ModelCheckpoint(legacy_model_path, monitor="val_loss", save_best_only=True, verbose=0),
        EarlyStopping(
            monitor="val_loss",
            patience=config.patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.patience_reduce_lr,
            min_lr=config.min_lr,
            verbose=1,
        ),
        CSVLogger(history_csv_path),
    ]

    # ── Class Weights ──
    from sklearn.utils.class_weight import compute_class_weight

    class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}
    logger.info(f"Class weights: {class_weights}")

    # ── Save Config ──
    config_dict = config.to_dict()
    config_dict["pretrained"] = pretrained or ""
    config_dict["resume"] = resume or ""
    config_dict["input_shape"] = list(input_shape)
    config_dict["num_classes"] = num_classes
    config_dict["train_samples_raw"] = int(len(X_train) / (6 if config.augment else 3))
    config_dict["train_samples_augmented"] = len(X_train)
    config_dict["val_samples"] = len(X_val)
    config_dict["test_samples"] = len(X_test)
    config_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config saved to: {config_path}")

    # ── Train ──
    logger.info("Starting training...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ──
    metrics = generate_evaluation(
        model, X_test, y_test, history, experiment_dir, TARGET_CLASSES
    )

    # ── Save Legacy Metadata ──
    metadata = {
        "classes": TARGET_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "input_shape": list(input_shape),
        "window_size": config.window_size,
        "num_keypoints": NUM_KEYPOINTS,
        "normalization": "divide_by_max_dim",
        "test_accuracy": metrics["test_accuracy"],
        "test_loss": metrics["test_loss"],
        "macro_f1": metrics["macro_avg"]["f1"],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "experiment_dir": experiment_dir,
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"  Experiment saved to: {experiment_dir}")
    logger.info(f"  Best model: {best_model_path}")
    logger.info(f"  Legacy model: {legacy_model_path}")
    logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"  Macro F1:     {metrics['macro_avg']['f1']:.4f}")
    logger.info("=" * 60)

    return model, history


# ─────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train / Fine-tune / Resume the Badminton Shot Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch on VideoBadminton dataset (broadcast features)
  python -m cv_model.train_model --dataset-dir path/to/VideoBadminton_Dataset --max-shots 300

  # Fine-tune from existing model with lower learning rate
  python -m cv_model.train_model --pretrained cv_model/models/shot_classifier_v2.keras --lr 5e-5 --epochs 50

  # Resume interrupted training
  python -m cv_model.train_model --resume cv_model/runs/experiment_001/last_model.keras

  # Train side-angle model from scratch using labeled videos
  python -m cv_model.train_model --mode side --epochs 100

  # Fine-tune side-angle model on newly labeled data
  python -m cv_model.train_model --mode side --pretrained cv_model/models/shot_classifier_side_v1.keras --lr 1e-4
""",
    )

    # ── Mode & Data ──
    parser.add_argument(
        "--mode", choices=["broadcast", "side"], default="broadcast",
        help="Feature mode: 'broadcast' (34-dim) or 'side' (42-dim). Default: broadcast",
    )
    parser.add_argument(
        "--dataset-dir", default=None,
        help="Path to VideoBadminton folder dataset (auto-loads from folder structure)",
    )
    parser.add_argument(
        "--dataset-folders", default=None,
        help="Comma-separated folder:class mappings, e.g., '14_Smash:smash' or '14_Smash:smash,05_Drop Shot:drop'",
    )
    parser.add_argument(
        "--video-dir", default=None,
        help="Directory containing match videos (auto-detected per mode)",
    )
    parser.add_argument("--cache", default=None, help="Cache file for extracted poses")
    parser.add_argument(
        "--merge-data", default=None,
        help="Comma-separated paths to additional .npz cache files to merge",
    )

    # ── Fine-Tuning ──
    parser.add_argument(
        "--pretrained", default=None,
        help="Path to pretrained .keras model for fine-tuning (loads weights, resets optimizer)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint .keras to resume training (preserves optimizer state)",
    )

    # ── Hyperparameters ──
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32)")
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw"], default=None,
        help="Optimizer. Default: adam (scratch) / adamw (fine-tune)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--max-shots", type=int, default=500, help="Max shots per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Use basic augmentation instead of advanced",
    )

    # ── Output ──
    parser.add_argument(
        "--output-dir", default="cv_model/models",
        help="Directory to save trained model (legacy path)",
    )

    args = parser.parse_args()

    # ── Build TrainingConfig ──
    is_finetuning = bool(args.pretrained or args.resume)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        seed=args.seed,
        weight_decay=args.weight_decay,
        augment=not args.no_augment,
        max_per_class=args.max_shots,
        mode=args.mode,
        pretrained=args.pretrained or "",
        resume=args.resume or "",
    )

    # Smart defaults: fine-tuning uses lower LR and AdamW
    if is_finetuning:
        config.lr = args.lr if args.lr is not None else 5e-5
        config.optimizer = args.optimizer or "adamw"
    else:
        config.lr = args.lr if args.lr is not None else 1e-3
        config.optimizer = args.optimizer or "adam"

    # ── Create experiment directory ──
    experiment_dir = get_next_experiment_dir()

    logger.info("═" * 60)
    if args.resume:
        logger.info("  MODE: RESUME TRAINING")
    elif args.pretrained:
        logger.info("  MODE: FINE-TUNING")
    else:
        logger.info("  MODE: TRAIN FROM SCRATCH")
    logger.info(f"  Feature mode: {args.mode}")
    logger.info(f"  Learning rate: {config.lr}")
    logger.info(f"  Optimizer: {config.optimizer}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Experiment: {experiment_dir}")
    logger.info("═" * 60)

    # ═════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═════════════════════════════════════════════════════════════════

    X, y = None, None

    if args.dataset_dir:
        # ── VIDEOBADMINTON DATASET MODE ──
        cache_path = args.cache or "cv_model/training_data/videobadminton_cache.npz"

        logger.info("DATA SOURCE: VideoBadminton folder dataset")
        logger.info(f"  Dataset dir: {args.dataset_dir}")

        if os.path.exists(cache_path):
            logger.info(f"  Loading cached data from {cache_path}")
            cache = np.load(cache_path)
            X, y = cache["X"], cache["y"]
            logger.info(f"  Loaded {len(X)} samples from cache")
        else:
            logger.info("  Loading clips from folder structure...")
            custom_map = None
            if args.dataset_folders:
                custom_map = {}
                for item in args.dataset_folders.split(","):
                    if ":" in item:
                        folder, cls_name = item.split(":", 1)
                        custom_map[folder.strip()] = cls_name.strip()
                    else:
                        logger.warning(f"Ignoring invalid folder mapping format (expected folder:class): '{item}'")
                logger.info(f"Using custom folder mapping: {custom_map}")

            clips = load_videobadminton_clips(
                args.dataset_dir, class_map=custom_map, max_per_class=config.max_per_class, seed=config.seed
            )
            if not clips:
                logger.error("No clips found! Check --dataset-dir path.")
                sys.exit(1)

            logger.info("  Extracting pose sequences (this may take a while)...")
            X, y = extract_videobadminton_sequences(
                clips, window_size=config.window_size, feature_mode=args.mode
            )
            if len(X) == 0:
                logger.error("No sequences extracted! Check video files.")
                sys.exit(1)

            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, X=X, y=y)
            logger.info(f"  Cached pose data to {cache_path}")

    elif args.mode == "side":
        # ── SIDE-ANGLE MODE ──
        video_dir = args.video_dir or SIDE_VIDEO_DIR
        cache_path = args.cache or "cv_model/training_data/side_pose_cache.npz"

        logger.info("DATA SOURCE: Side-angle labeled videos")

        if os.path.exists(cache_path):
            logger.info(f"  Loading cached data from {cache_path}")
            cache = np.load(cache_path)
            X, y = cache["X"], cache["y"]
            logger.info(f"  Loaded {len(X)} samples from cache")
        else:
            samples = load_side_angle_annotations(video_dir=video_dir)
            if not samples:
                logger.error("No samples found! Label videos first: python -m cv_model.label_tool <video>")
                sys.exit(1)

            logger.info("  Extracting pose sequences from side-angle videos...")
            X, y, valid_samples = extract_side_pose_sequences(samples, video_dir, window_size=config.window_size)
            if len(X) == 0:
                logger.error("No pose sequences extracted!")
                sys.exit(1)

            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, X=X, y=y)
            logger.info(f"  Cached pose data to {cache_path}")

    else:
        # ── BROADCAST MODE (ShuttleSet) ──
        video_dir = args.video_dir or "cv_model/training_data/videos"
        cache_path = args.cache or "cv_model/training_data/pose_cache.npz"

        logger.info("DATA SOURCE: ShuttleSet broadcast annotations")

        if os.path.exists(cache_path):
            logger.info(f"  Loading cached data from {cache_path}")
            cache = np.load(cache_path)
            X, y = cache["X"], cache["y"]
            logger.info(f"  Loaded {len(X)} samples from cache")
        else:
            samples = load_shuttleset_annotations(max_per_class=config.max_per_class, video_dir=video_dir)
            if not samples:
                logger.error("No samples found! Check ShuttleSet CSVs.")
                sys.exit(1)

            logger.info("  Extracting pose sequences from videos...")
            X, y, valid_samples = extract_pose_sequences(
                samples, video_dir, MATCH_CSV, window_size=config.window_size
            )
            if len(X) == 0:
                logger.error("No pose sequences extracted!")
                sys.exit(1)

            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, X=X, y=y)
            logger.info(f"  Cached pose data to {cache_path}")

    # ── Merge additional data if requested ──
    if args.merge_data:
        for extra_path in args.merge_data.split(","):
            extra_path = extra_path.strip()
            if os.path.exists(extra_path):
                extra = np.load(extra_path)
                X_extra, y_extra = extra["X"], extra["y"]
                logger.info(f"  Merging {len(X_extra)} samples from {extra_path}")
                X = np.concatenate([X, X_extra], axis=0)
                y = np.concatenate([y, y_extra], axis=0)
            else:
                logger.warning(f"  Merge file not found: {extra_path}")

    # ── Final data summary ──
    logger.info(f"Total dataset: {len(X)} samples, X shape={X.shape}")
    unique_classes = np.unique(y)
    for cls_name, cls_id in CLASS_TO_ID.items():
        logger.info(f"  {cls_name}: {np.sum(y == cls_id)}")

    if len(unique_classes) < 2:
        logger.info("=" * 60)
        logger.info("NOTE: Less than 2 unique classes were loaded/extracted.")
        logger.info(f"Cached poses saved to: {cache_path}")
        logger.info("Exiting gracefully without training. To train the model, you need at least 2 classes.")
        logger.info("You can merge multiple single-class cache files during training using:")
        logger.info("  --merge-data cache1.npz,cache2.npz")
        logger.info("=" * 60)
        return

    # ═════════════════════════════════════════════════════════════════
    # TRAINING
    # ═════════════════════════════════════════════════════════════════

    model, history = train_model(
        X, y,
        config=config,
        pretrained=args.pretrained,
        resume=args.resume,
        experiment_dir=experiment_dir,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
