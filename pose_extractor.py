#!/usr/bin/env python3
# ABOUTME: Extracts player body poses from video frames using YOLO pose estimation.
# ABOUTME: Returns normalized 13-keypoint sequences suitable for LSTM shot classification.
"""
Pose Extractor Module
=====================
Uses Ultralytics YOLO-Pose to detect and track players in badminton video,
extracting 13 body keypoints per frame (eyes/ears excluded to match LSTM input).

The output is a time-series of keypoint coordinates ready for the LSTM classifier.
"""

import cv2
import numpy as np
import logging
import math
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# YOLO-Pose outputs 17 COCO keypoints in this order:
YOLO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# The 13 keypoints the LSTM model was trained on (eyes and ears excluded)
MODEL_KEYPOINTS = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# Indices into the 17-keypoint YOLO output for the 13 we need
MODEL_KEYPOINT_INDICES = [
    YOLO_KEYPOINT_NAMES.index(name) for name in MODEL_KEYPOINTS
]

# Minimum confidence for a keypoint to be considered valid
MIN_KEYPOINT_CONFIDENCE = 0.3

# Default YOLO model for pose estimation
# Using nano model for speed; switch to yolo11x-pose.pt for max accuracy
DEFAULT_POSE_MODEL = "yolo11n-pose.pt"

# Max input resolution for YOLO (frames are downscaled to this before inference)
# YOLO internally resizes to 640 anyway, so feeding 4K frames wastes time on resizing
DEFAULT_MAX_INPUT_SIZE = 640


# ─────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FramePose:
    """Pose data for a single person in a single frame."""
    frame_num: int
    person_id: int
    keypoints_xy: np.ndarray          # (13, 2) array of (x, y)
    keypoints_conf: np.ndarray        # (13,) confidence scores
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    bbox_conf: float

    @property
    def center_y(self) -> float:
        """Vertical center of the bounding box (useful for top/bottom player)."""
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def center_x(self) -> float:
        """Horizontal center of the bounding box."""
        return (self.bbox[0] + self.bbox[2]) / 2.0


@dataclass
class PlayerSequence:
    """Time-series pose sequence for one player across many frames."""
    player_id: str                        # e.g. "Player_A"
    frames: List[FramePose] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def to_numpy(self) -> np.ndarray:
        """
        Convert to a (T, 13, 2) numpy array where T = number of frames.
        Missing keypoints are filled with 0.0.
        """
        if not self.frames:
            return np.zeros((0, 13, 2))
        arr = np.stack([f.keypoints_xy for f in self.frames], axis=0)
        return arr


# ─────────────────────────────────────────────────────────────────────
# Pose Extractor
# ─────────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    Extracts player poses from badminton video using YOLO-Pose.

    Usage:
        extractor = PoseExtractor()
        players = extractor.extract("match.mp4", max_frames=120)
        # players["Player_A"].to_numpy()  →  (T, 13, 2) array
    """

    def __init__(self, model_path: str = DEFAULT_POSE_MODEL, device: str = "auto"):
        """
        Args:
            model_path: Path to YOLO-Pose weights (downloaded automatically).
            device: "cuda", "cpu", or "auto" (auto-detect GPU).
        """
        self.model_path = model_path
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return

        from ultralytics import YOLO
        import torch

        logger.info(f"Loading YOLO-Pose model: {self.model_path}")
        self._model = YOLO(self.model_path)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            self._model.to("cuda")
            logger.info(f"YOLO running on CUDA: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("YOLO running on CPU")

    def extract(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        max_players: int = 2,
        normalize: bool = True,
        frame_skip: int = 1,
        max_input_size: int = DEFAULT_MAX_INPUT_SIZE,
        crop_region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, PlayerSequence]:
        """
        Extract pose sequences for up to `max_players` from a video.

        Args:
            video_path:     Path to the video file.
            start_frame:    First frame to process (inclusive).
            end_frame:      Last frame to process (inclusive). None = all.
            max_players:    Maximum number of players to track (default 2).
            normalize:      If True, normalize coordinates to [0, 1] range.
            frame_skip:     Process every Nth frame (1 = all, 2 = every other, 3 = every 3rd).
                            Skipped frames are interpolated. Higher = faster but less smooth.
            max_input_size: Downscale frames so the longest side is at most this many pixels
                            before sending to YOLO. Lower = faster. Default 640.
            crop_region:    Crop the video frame before processing. Given as fractional
                            coordinates (left, top, right, bottom) in [0.0, 1.0].
                            Example: (0.0, 0.0, 0.5, 1.0) = left half of frame.
                                     (0.5, 0.0, 1.0, 1.0) = right half of frame.

        Returns:
            Dict mapping player IDs ("Player_A", "Player_B", ...) to
            their PlayerSequence objects.
        """
        self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if end_frame is None:
            end_frame = total_frames - 1

        # ── Apply crop region ──
        if crop_region:
            cl, ct, cr, cb = crop_region
            crop_x1 = int(cl * frame_w)
            crop_y1 = int(ct * frame_h)
            crop_x2 = int(cr * frame_w)
            crop_y2 = int(cb * frame_h)
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            logger.info(
                f"Cropping frame to region ({cl:.0%},{ct:.0%})-({cr:.0%},{cb:.0%}) "
                f"= {crop_w}x{crop_h}px"
            )
        else:
            crop_x1, crop_y1 = 0, 0
            crop_w, crop_h = frame_w, frame_h

        # ── Compute downscale factor (applied AFTER cropping) ──
        scale = 1.0
        longest_side = max(crop_w, crop_h)
        if longest_side > max_input_size:
            scale = max_input_size / longest_side
            scaled_w = int(crop_w * scale)
            scaled_h = int(crop_h * scale)
            logger.info(f"Downscaling {crop_w}x{crop_h} → {scaled_w}x{scaled_h} for YOLO")
        else:
            scaled_w, scaled_h = crop_w, crop_h

        total_to_process = (end_frame - start_frame + 1)
        frames_to_run = (total_to_process + frame_skip - 1) // frame_skip

        logger.info(
            f"Video: {video_path} | {frame_w}x{frame_h} @ {fps:.1f}fps | "
            f"Frames {start_frame}→{end_frame} ({total_to_process} total, "
            f"processing every {frame_skip} → {frames_to_run} YOLO runs)"
        )

        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # ── Track detections per person across frames ──
        # We use a custom distance-based tracker to prevent jumping to hallucinations
        track_id_frames: Dict[int, List[FramePose]] = {}
        last_positions: Dict[int, Tuple[float, float]] = {}  # p_idx -> (center_x, center_y)

        frame_num = start_frame
        processed = 0
        yolo_runs = 0

        while frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Frame skipping: only run YOLO on every Nth frame ──
            if (frame_num - start_frame) % frame_skip != 0:
                frame_num += 1
                continue

            # ── Crop the frame if needed ──
            if crop_region:
                work_frame = frame[crop_y1:crop_y1+crop_h, crop_x1:crop_x1+crop_w]
            else:
                work_frame = frame

            # ── Downscale for speed ──
            if scale < 1.0:
                small_frame = cv2.resize(work_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            else:
                small_frame = work_frame

            # Run standard YOLO-Pose (no tracker needed, we do spatial assignment)
            results = self._model(
                small_frame, verbose=False,
                conf=0.3, iou=0.5
            )

            if results and results[0].keypoints is not None:
                result = results[0]
                keypoints_data = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
                boxes = result.boxes

                # Collect all valid poses in this frame
                frame_poses = []
                for i in range(len(keypoints_data)):
                    # Extract the 13 model keypoints
                    kps_17 = keypoints_data[i]  # (17, 3) = x, y, conf
                    kps_13_xy = kps_17[MODEL_KEYPOINT_INDICES, :2]     # (13, 2)
                    kps_13_conf = kps_17[MODEL_KEYPOINT_INDICES, 2]    # (13,)

                    # Scale coordinates back to cropped region resolution
                    if scale < 1.0:
                        kps_13_xy[:, 0] /= scale
                        kps_13_xy[:, 1] /= scale

                    # Offset coordinates back to full-frame position
                    if crop_region:
                        kps_13_xy[:, 0] += crop_x1
                        kps_13_xy[:, 1] += crop_y1

                    # Zero out low-confidence keypoints
                    low_conf_mask = kps_13_conf < MIN_KEYPOINT_CONFIDENCE
                    kps_13_xy[low_conf_mask] = 0.0

                    # Normalize to [0, 1] range while PRESERVING aspect ratio
                    if normalize:
                        max_dim = float(max(frame_w, frame_h))
                        kps_13_xy[:, 0] /= max_dim
                        kps_13_xy[:, 1] /= max_dim

                    # Bounding box (scale back and offset to original frame)
                    box_xyxy = boxes.xyxy[i].cpu().numpy()
                    if scale < 1.0:
                        box_xyxy /= scale
                    if crop_region:
                        box_xyxy[0] += crop_x1
                        box_xyxy[1] += crop_y1
                        box_xyxy[2] += crop_x1
                        box_xyxy[3] += crop_y1
                    box_xyxy = box_xyxy.astype(int)
                    box_conf = float(boxes.conf[i].item())

                    fp = FramePose(
                        frame_num=frame_num,
                        person_id=i,  # Temporary ID
                        keypoints_xy=kps_13_xy.copy(),
                        keypoints_conf=kps_13_conf.copy(),
                        bbox=tuple(box_xyxy),
                        bbox_conf=box_conf,
                    )
                    frame_poses.append(fp)

                # ── Robust Dynamic Tracking (Movement-Based) ──
                # Assign to existing tracks if close enough, else spawn new track
                next_track_id = max(track_id_frames.keys()) + 1 if track_id_frames else 0
                
                unmatched_poses = frame_poses.copy()
                
                # Try matching existing tracks first
                for p_idx, last_pos in last_positions.items():
                    if not unmatched_poses:
                        break
                        
                    def _dist(p: FramePose) -> float:
                        dx = (p.center_x - last_pos[0]) / frame_w
                        dy = (p.center_y - last_pos[1]) / frame_h
                        return dx*dx + dy*dy
                        
                    unmatched_poses.sort(key=_dist)
                    best_match = unmatched_poses[0]
                    
                    # 0.02 squared distance = ~14% of screen jump in a single frame
                    if _dist(best_match) < 0.02:
                        best_match.person_id = p_idx
                        track_id_frames[p_idx].append(best_match)
                        last_positions[p_idx] = (best_match.center_x, best_match.center_y)
                        unmatched_poses.pop(0)
                
                # Any remaining unmatched poses get a new track ID
                for p in unmatched_poses:
                    # Ignore really low confidence ghosts before creating new tracks
                    if p.bbox_conf > 0.4:
                        p.person_id = next_track_id
                        track_id_frames[next_track_id] = [p]
                        last_positions[next_track_id] = (p.center_x, p.center_y)
                        next_track_id += 1

            frame_num += 1
            processed += 1
            yolo_runs += 1

            if yolo_runs % 50 == 0:
                pct = yolo_runs / frames_to_run * 100
                logger.info(f"Pose extraction: {yolo_runs}/{frames_to_run} YOLO runs ({pct:.0f}%)")

        cap.release()
        
        # ── Movement Scoring and Filtering ──
        # Calculate total movement for each track to identify the real players
        track_scores = []
        for p_idx, frames in track_id_frames.items():
            # Calculate duration in seconds, accounting for frame_skip
            track_duration = (len(frames) * frame_skip) / fps
            
            # Skip tracks that are very short (less than 1.0 seconds, or 30% of short videos)
            min_duration = min(1.0, (total_frames / fps) * 0.3)
            
            if track_duration < min_duration:
                continue
                
            movement_score = 0.0
            for i in range(1, len(frames)):
                dx = frames[i].center_x - frames[i-1].center_x
                dy = frames[i].center_y - frames[i-1].center_y
                movement_score += math.sqrt(dx*dx + dy*dy)
                
            track_scores.append((p_idx, movement_score, frames))
            
        # Sort by movement score descending
        track_scores.sort(key=lambda x: x[1], reverse=True)
        top_tracks = track_scores[:max_players]
        
        # Sort top tracks left-to-right (for broadcast consistency)
        if len(top_tracks) > 1:
            top_tracks.sort(key=lambda x: x[2][0].center_x)
            
        # ── Format output ──
        players: Dict[str, PlayerSequence] = {}
        player_labels = ["Player_A", "Player_B", "Player_C", "Player_D"]
        
        for i, (old_p_idx, score, frames) in enumerate(top_tracks):
            label = player_labels[i] if i < len(player_labels) else f"Player_{i}"
            for f in frames:
                f.person_id = i # Rewrite ID to 0/1
            ps = PlayerSequence(player_id=label, frames=frames)
            players[label] = ps
            
        logger.info(
            f"Pose extraction complete. {yolo_runs} YOLO runs "
            f"(skipped {total_to_process - yolo_runs} frames), "
            f"tracked {len(players)} players"
        )
        return players

    def extract_single_frame(self, frame: np.ndarray, normalize: bool = True) -> List[FramePose]:
        """
        Extract poses from a single BGR frame (useful for live camera feeds).

        Args:
            frame:     BGR image as numpy array.
            normalize: If True, normalize coordinates to [0, 1].

        Returns:
            List of FramePose objects (one per detected person).
        """
        self._load_model()

        frame_h, frame_w = frame.shape[:2]
        results = self._model(frame, verbose=False, conf=0.4)

        poses = []
        if results and results[0].keypoints is not None:
            result = results[0]
            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for i in range(len(keypoints_data)):
                kps_17 = keypoints_data[i]
                kps_13_xy = kps_17[MODEL_KEYPOINT_INDICES, :2]
                kps_13_conf = kps_17[MODEL_KEYPOINT_INDICES, 2]

                low_conf_mask = kps_13_conf < MIN_KEYPOINT_CONFIDENCE
                kps_13_xy[low_conf_mask] = 0.0

                if normalize:
                    max_dim = float(max(frame_w, frame_h))
                    kps_13_xy[:, 0] /= max_dim
                    kps_13_xy[:, 1] /= max_dim

                box_xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                box_conf = float(boxes.conf[i].item())

                fp = FramePose(
                    frame_num=0,
                    person_id=i,
                    keypoints_xy=kps_13_xy.copy(),
                    keypoints_conf=kps_13_conf.copy(),
                    bbox=tuple(box_xyxy),
                    bbox_conf=box_conf,
                )
                poses.append(fp)

        return poses
