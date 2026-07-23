#!/usr/bin/env python3
# ABOUTME: Inference engine that ties YOLO pose extraction to LSTM shot classification.
# ABOUTME: Provides video overlay, live camera, and batch prediction APIs.
"""
Inference Engine
================
High-level API for predicting badminton shot types from video or camera feeds.

Combines:
  - PoseExtractor (YOLO-Pose → 13 keypoints per frame)
  - Pre-trained LSTM model (keypoint sequences → shot class)

Outputs predictions with confidence scores and optional video overlays
showing the detected skeleton and predicted shot label.
"""

import os
import math
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from cv_model.pose_extractor import PoseExtractor, PlayerSequence, FramePose, MODEL_KEYPOINTS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

SHOT_CLASSES = ['clear', 'drop', 'smash']

# ─────────────────────────────────────────────────────────────────────
# Biomechanical Shot Detection Gate
# ─────────────────────────────────────────────────────────────────────
# A real badminton stroke (smash, clear, drop) requires a fast arm swing.
# These thresholds are in normalized-coordinate units per second.  With
# max-dim normalization the full frame diagonal is ~1.0, and a player's
# arm span is roughly 0.3–0.5 of the frame height.  During a real stroke
# the dominant wrist easily moves at 2–5 units/sec (normalized) and
# accelerates at 30+ units/sec².  During footwork / walking the wrist
# drifts at < 0.8 units/sec.  We set conservative thresholds so that
# any genuine stroke passes but idle movement is rejected.
#
# SWING_SPEED_THRESHOLD: minimum peak wrist speed (units/sec) in the
#   window for us to trust the LSTM's prediction.  Below this we assume
#   the player is not hitting.
# SWING_ACCEL_THRESHOLD: minimum peak wrist acceleration (units/sec²).
#   Explosive strokes always have a sharp velocity spike even if the
#   peak speed is modest (e.g. a soft drop from a high preparation).
#
# A window must exceed AT LEAST ONE of these two thresholds to be
# classified as a shot.  If neither is exceeded the window is labeled
# "no_shot".
SWING_SPEED_THRESHOLD = 1.2    # units/sec (normalized coords)
SWING_ACCEL_THRESHOLD = 15.0   # units/sec²

# Skeleton connections for drawing (indices into the 13-keypoint array)
# Mapping: 0=nose, 1=L_shoulder, 2=R_shoulder, 3=L_elbow, 4=R_elbow,
#          5=L_wrist, 6=R_wrist, 7=L_hip, 8=R_hip, 9=L_knee, 10=R_knee,
#          11=L_ankle, 12=R_ankle
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),       # nose → shoulders
    (1, 3), (3, 5),       # left arm
    (2, 4), (4, 6),       # right arm
    (1, 7), (2, 8),       # torso
    (7, 8),               # hips
    (7, 9), (9, 11),      # left leg
    (8, 10), (10, 12),    # right leg
]

# Colors for drawing (BGR)
SKELETON_COLOR = (0, 255, 170)      # green-cyan
KEYPOINT_COLOR = (0, 200, 255)      # orange
LABEL_BG_COLOR = (30, 30, 30)       # dark grey
LABEL_TEXT_COLOR = (255, 255, 255)   # white

# Default path to LSTM weights (relative to this file)
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "temp_badminton_repo" / "weights"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "shot_classifier_v2.keras"
DEFAULT_SIDE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "shot_classifier_side_v1.keras"


# ─────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ShotPrediction:
    """Result of a shot classification."""
    shot_type: str              # e.g. "smash" or "no_shot"
    confidence: float           # 0.0 – 1.0
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    player_id: str = ""
    frame_range: Tuple[int, int] = (0, 0)
    is_shot: bool = True        # False when the biomechanical gate rejected this window

    def __repr__(self):
        tag = '' if self.is_shot else ' [NO SHOT]'
        return (
            f"ShotPrediction('{self.shot_type}' @ {self.confidence:.1%}, "
            f"player='{self.player_id}', frames={self.frame_range}{tag})"
        )


# ─────────────────────────────────────────────────────────────────────
# LSTM Model Wrapper
# ─────────────────────────────────────────────────────────────────────

class LSTMShotModel:
    """
    Wraps the pre-trained LSTM Keras model for shot prediction.

    Handles Keras 2 → 3 compatibility (time_major removal) automatically.
    """

    def __init__(self, model_path: str = None, model_type: str = "broadcast"):
        self.model_type = model_type
        if model_path:
            self.model_path = model_path
        elif model_type == "side":
            self.model_path = str(DEFAULT_SIDE_MODEL_PATH)
        else:
            self.model_path = str(DEFAULT_MODEL_PATH)
        self._model = None

    def _load_model(self):
        """Lazy-load the Keras model with backward compatibility."""
        if self._model is not None:
            return

        logger.info(f"Loading LSTM model from: {self.model_path}")

        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.layers import LSTM as LSTM_Layer

        # Keras 3 compatibility: strip 'time_major' from older models
        class LegacyLSTM(LSTM_Layer):
            def __init__(self, *args, **kwargs):
                kwargs.pop('time_major', None)
                super().__init__(*args, **kwargs)

        try:
            self._model = load_model(self.model_path, compile=False)
        except (ValueError, TypeError) as e:
            logger.warning(f"Standard load failed ({e}), trying legacy workaround...")
            self._model = load_model(
                self.model_path, compile=False,
                custom_objects={'LSTM': LegacyLSTM}
            )

        logger.info("LSTM model loaded successfully.")

    @staticmethod
    def _check_swing_gate(keypoint_sequence: np.ndarray, frame_skip: int = 2,
                          fps: float = 30.0) -> Tuple[bool, float, float]:
        """
        Biomechanical gate: check whether the keypoint window contains a
        genuine arm swing by measuring peak wrist speed and acceleration.

        Returns:
            (is_shot, peak_speed, peak_accel)
        """
        from cv_model.features import R_WRI, L_WRI, _is_valid

        T = keypoint_sequence.shape[0]
        dt = frame_skip / fps

        # Compute wrist speeds for both hands
        speeds = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            best_speed = 0.0
            for wri_idx in (R_WRI, L_WRI):
                curr = keypoint_sequence[t, wri_idx]
                prev = keypoint_sequence[t - 1, wri_idx]
                if _is_valid(curr) and _is_valid(prev):
                    dx = float(curr[0] - prev[0]) / dt
                    dy = float(curr[1] - prev[1]) / dt
                    spd = math.hypot(dx, dy)
                    best_speed = max(best_speed, spd)
            speeds[t] = best_speed

        # Compute acceleration (derivative of speed)
        accels = np.zeros(T, dtype=np.float32)
        for t in range(2, T):
            accels[t] = abs(speeds[t] - speeds[t - 1]) / dt

        peak_speed = float(np.max(speeds))
        peak_accel = float(np.max(accels))

        is_shot = (peak_speed >= SWING_SPEED_THRESHOLD or
                   peak_accel >= SWING_ACCEL_THRESHOLD)

        return is_shot, peak_speed, peak_accel

    def predict(self, keypoint_sequence: np.ndarray,
                frame_skip: int = 2, fps: float = 30.0) -> ShotPrediction:
        """
        Predict the shot type from a keypoint sequence.

        Applies a biomechanical gate first: if the wrist speed and
        acceleration are both below threshold (no arm swing detected),
        the window is classified as "no_shot" without even querying
        the LSTM, preventing false positives during footwork/idle.

        Args:
            keypoint_sequence: numpy array of shape (T, 13, 2) where
                               T = number of frames.
            frame_skip: frames skipped between entries (for velocity calc).
            fps: source video frame rate.

        Returns:
            ShotPrediction with shot_type, confidence, and full probabilities.
        """
        # ── Biomechanical gate ──
        is_shot, peak_speed, peak_accel = self._check_swing_gate(
            keypoint_sequence, frame_skip=frame_skip, fps=fps
        )

        if not is_shot:
            return ShotPrediction(
                shot_type="no_shot",
                confidence=0.0,
                all_probabilities={name: 0.0 for name in SHOT_CLASSES},
                is_shot=False,
            )

        # ── LSTM prediction (only reached when a swing is detected) ──
        self._load_model()

        if self.model_type == "side":
            from cv_model.features import compute_side_angle_features
            enhanced_seq = compute_side_angle_features(keypoint_sequence,
                                                       fps=fps, frame_skip=frame_skip)
        else:
            from cv_model.features import compute_kinematic_features
            enhanced_seq = compute_kinematic_features(keypoint_sequence,
                                                      fps=fps, frame_skip=frame_skip)

        # Batch dimension: (1, T, features)
        flat = np.expand_dims(enhanced_seq, axis=0)

        # Run prediction
        probs = self._model.predict(flat, verbose=0)  # (1, num_classes)
        probs = probs[0]

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        shot_type = SHOT_CLASSES[predicted_idx]

        all_probs = {name: float(probs[i]) for i, name in enumerate(SHOT_CLASSES)}

        return ShotPrediction(
            shot_type=shot_type,
            confidence=confidence,
            all_probabilities=all_probs,
            is_shot=True,
        )


# ─────────────────────────────────────────────────────────────────────
# High-Level Predictor
# ─────────────────────────────────────────────────────────────────────

class ShotPredictor:
    """
    End-to-end shot predictor: Video → Pose → LSTM → Shot Type.

    Usage:
        predictor = ShotPredictor()

        # From video file
        results = predictor.predict_from_video("match.mp4")

        # With annotated output video
        results = predictor.predict_from_video(
            "match.mp4", output_video="annotated.mp4"
        )

        # From live camera
        predictor.predict_live(camera_index=0)
    """

    def __init__(
        self,
        model_path: str = None,
        model_type: str = "broadcast",
        yolo_model: str = "yolo11n-pose.pt",
        device: str = "auto",
    ):
        self.pose_extractor = PoseExtractor(model_path=yolo_model, device=device)
        self.lstm_model = LSTMShotModel(model_path=model_path, model_type=model_type)

    def predict_from_video(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        output_video: Optional[str] = None,
        window_size: int = 41,
        stride: int = 5,
        frame_skip: int = 2,
        crop_region: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[ShotPrediction]:
        """
        Predict shot types from a video file.

        Uses a sliding window approach: the video is split into overlapping
        windows of `window_size` frames, each classified independently.

        Args:
            video_path:    Path to the input video.
            start_frame:   First frame (inclusive).
            end_frame:     Last frame (inclusive). None = entire video.
            output_video:  If set, writes an annotated video with skeleton + label.
            window_size:   Number of frames per classification window (default 41).
            stride:        Step between windows (default 20).
            frame_skip:    Process every Nth frame for speed (default 2).
            crop_region:   Crop the frame as (left, top, right, bottom) in [0, 1].
                           Use 'left'=(0,0,0.5,1) or 'right'=(0.5,0,1,1) for
                           side-by-side videos.

        Returns:
            List of ShotPrediction objects, one per window.
        """
        logger.info(f"═══ Predicting shots from: {video_path} ═══")

        # Step 1: Extract poses
        # Always track at least 2 people to ensure we capture both the active player 
        # and any static background people (umpire/coach), then we filter by movement.
        max_players = 2
        players = self.pose_extractor.extract(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_skip=frame_skip,
            crop_region=crop_region,
            max_players=max_players,
        )

        if not players:
            logger.warning("No players detected in video.")
            return []

        # Find the player with the most physical movement (variance in position)
        # This prevents picking a static umpire who might have high YOLO confidence
        def get_movement_score(p_seq):
            if len(p_seq.frames) < 10:
                return 0.0
            xs = [p.center_x for p in p_seq.frames]
            ys = [p.center_y for p in p_seq.frames]
            return float(np.std(xs) + np.std(ys))

        primary_player_id = max(players, key=lambda k: get_movement_score(players[k]))
        player = players[primary_player_id]
        logger.info(f"Primary player: {primary_player_id} (score: {get_movement_score(player):.2f}, {player.num_frames} frames)")

        # Step 2: Get video fps for biomechanical calculations
        _cap = cv2.VideoCapture(video_path)
        video_fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
        _cap.release()

        # Step 3: Sliding window classification with biomechanical gate
        sequence = player.to_numpy()  # (T, 13, 2)
        T = sequence.shape[0]
        all_predictions: List[ShotPrediction] = []

        if T < window_size:
            # Pad shorter sequences
            padded = np.zeros((window_size, 13, 2))
            padded[:T] = sequence
            pred = self.lstm_model.predict(padded, frame_skip=frame_skip, fps=video_fps)
            pred.player_id = primary_player_id
            pred.frame_range = (start_frame, start_frame + (T - 1) * frame_skip)
            all_predictions.append(pred)
        else:
            # Sliding window
            for w_start in range(0, T - window_size + 1, stride):
                window = sequence[w_start:w_start + window_size]
                pred = self.lstm_model.predict(window, frame_skip=frame_skip, fps=video_fps)
                pred.player_id = primary_player_id
                pred.frame_range = (
                    start_frame + (w_start * frame_skip),
                    start_frame + ((w_start + window_size - 1) * frame_skip)
                )
                all_predictions.append(pred)

        # Separate real shot detections from idle/footwork windows
        predictions = [p for p in all_predictions if p.is_shot]
        suppressed = [p for p in all_predictions if not p.is_shot]

        # Log results to console
        if suppressed:
            logger.info(f"  Biomechanical gate suppressed {len(suppressed)}/{len(all_predictions)} "
                        f"windows (no arm swing detected)")
        if not predictions:
            logger.info("  No shots detected in this video (all windows below swing threshold).")
        for p in predictions:
            logger.info(f"  Frames {p.frame_range}: {p.shot_type.upper()} ({p.confidence:.1%})")

        # Step 3: Optionally write annotated output video
        if output_video:
            self._write_annotated_video(
                video_path, output_video, players, predictions,
                start_frame, end_frame, window_size
            )

        return predictions

    def predict_live(
        self,
        camera_index: int = 0,
        window_size: int = 41,
        display_width: int = 1280,
        display_height: int = 720,
    ):
        """
        Run live shot prediction from a camera feed.

        Collects frames in a rolling buffer, runs classification every
        `window_size` frames, and displays the skeleton + predicted shot
        in real-time. Press 'q' to quit.

        Args:
            camera_index:   Camera device index (default 0).
            window_size:    Number of frames per classification window.
            display_width:  Display window width.
            display_height: Display window height.
        """
        logger.info(f"Starting live prediction from camera {camera_index}")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

        frame_buffer: List[np.ndarray] = []   # rolling buffer of (13, 2) arrays
        current_prediction: Optional[ShotPrediction] = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]

            # Extract poses from this frame
            # We must use absolute pixels, NOT normalized coordinates, because the original model
            # was fed absolute pixels during inference.
            poses = self.pose_extractor.extract_single_frame(frame, normalize=False)

            if poses:
                # Use the person closest to center (likely the playing player)
                best_pose = min(poses, key=lambda p: abs(p.center_x - 0.5))
                frame_buffer.append(best_pose.keypoints_xy.copy())

                # Draw skeleton on frame
                self._draw_skeleton(frame, best_pose, frame_w, frame_h)
            else:
                # No detection: add zeros
                frame_buffer.append(np.zeros((13, 2)))

            # Keep only last `window_size` frames
            if len(frame_buffer) > window_size:
                frame_buffer.pop(0)

            # Run prediction when we have enough frames
            if len(frame_buffer) == window_size:
                seq = np.array(frame_buffer)  # (window_size, 13, 2)
                current_prediction = self.lstm_model.predict(seq)
                frame_count += 1

            # Draw prediction label
            if current_prediction:
                self._draw_label(frame, current_prediction)

            cv2.imshow("MotionIQ - Shot Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Live prediction ended.")

    # ─────────────────────────────────────────────────────────────────
    # Drawing Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_skeleton(
        frame: np.ndarray,
        pose: FramePose,
        frame_w: int,
        frame_h: int,
    ):
        """Draw the 13-keypoint skeleton on a frame."""
        kps = pose.keypoints_xy.copy()

        # De-normalize coordinates
        pts = np.zeros_like(kps, dtype=int)
        pts[:, 0] = (kps[:, 0] * frame_w).astype(int)
        pts[:, 1] = (kps[:, 1] * frame_h).astype(int)

        # Draw connections
        for (a, b) in SKELETON_CONNECTIONS:
            if np.all(kps[a] > 0) and np.all(kps[b] > 0):
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), SKELETON_COLOR, 2, cv2.LINE_AA)

        # Draw keypoints
        for i in range(13):
            if np.all(kps[i] > 0):
                cv2.circle(frame, tuple(pts[i]), 4, KEYPOINT_COLOR, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_label(frame: np.ndarray, prediction: ShotPrediction):
        """Draw the shot prediction label as an overlay."""
        label = f"{prediction.shot_type.upper()}: {prediction.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position: top-left
        x, y = 20, 50
        padding = 10

        # Background rectangle
        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            LABEL_BG_COLOR, -1
        )

        # Colored accent bar on the left
        accent_color = _shot_color(prediction.shot_type)
        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x - padding + 4, y + baseline + padding),
            accent_color, -1
        )

        # Text
        cv2.putText(frame, label, (x, y), font, font_scale, LABEL_TEXT_COLOR, thickness, cv2.LINE_AA)

    def _write_annotated_video(
        self,
        input_video: str,
        output_video: str,
        players: Dict[str, PlayerSequence],
        predictions: List[ShotPrediction],
        start_frame: int,
        end_frame: Optional[int],
        window_size: int,
    ):
        """Write an annotated video with skeleton overlays and shot labels."""
        logger.info(f"Writing annotated video: {output_video}")

        cap = cv2.VideoCapture(input_video)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total - 1

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

        # Build frame → FramePose lookup for all players
        frame_pose_lookup: Dict[int, FramePose] = {}
        for player in players.values():
            for fp in player.frames:
                # Use the primary player's poses
                if player.player_id == predictions[0].player_id if predictions else True:
                    frame_pose_lookup[fp.frame_num] = fp

        # Build a sorted list of predictions by their start frame.
        # On each frame, we show the prediction whose window contains that frame.
        # If a frame falls outside all windows, we show the LATEST past prediction
        # so the label never disappears.
        sorted_preds = sorted(predictions, key=lambda p: p.frame_range[0])

        def _get_prediction_for_frame(fnum: int) -> Optional[ShotPrediction]:
            """Return the best prediction for a given frame number by finding the closest window center."""
            best_pred = None
            min_dist = float('inf')
            
            for pred in sorted_preds:
                # Find the center frame of this prediction's window
                center = (pred.frame_range[0] + pred.frame_range[1]) / 2.0
                dist = abs(fnum - center)
                
                if dist < min_dist:
                    min_dist = dist
                    best_pred = pred
                    
            return best_pred

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fnum in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw skeleton
            if fnum in frame_pose_lookup:
                self._draw_skeleton(frame, frame_pose_lookup[fnum], frame_w, frame_h)

            # Draw prediction label (always shows latest prediction)
            pred_for_frame = _get_prediction_for_frame(fnum)
            if pred_for_frame:
                self._draw_label(frame, pred_for_frame)

            out.write(frame)

        cap.release()
        out.release()
        logger.info(f"Annotated video saved: {output_video}")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _shot_color(shot_type: str) -> Tuple[int, int, int]:
    """Return a distinct BGR color for each shot type."""
    colors = {
        'clear':   (255, 200, 0),     # cyan-ish
        'drive':   (0, 200, 255),     # orange
        'drop':    (0, 255, 100),     # green
        'lob':     (255, 100, 255),   # magenta
        'net':     (100, 255, 255),   # yellow
        'smash':   (0, 0, 255),       # red
        'no_shot': (120, 120, 120),   # grey (idle / footwork)
    }
    return colors.get(shot_type, (200, 200, 200))


# ─────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────

def main():
    """Command-line interface for shot prediction."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='MotionIQ Shot Predictor – Classify badminton shots from video'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # ── video subcommand ──
    vid_parser = subparsers.add_parser('video', help='Predict from a video file')
    vid_parser.add_argument('video_path', help='Path to the video file')
    vid_parser.add_argument('--model', default=None, help='Path to LSTM model (.h5 or .keras)')
    vid_parser.add_argument('--model-type', choices=['broadcast', 'side'], default='broadcast',
                            help="Model type: 'broadcast' (back-of-court) or 'side' (side-angle)")
    vid_parser.add_argument('--output', '-o', default=None, help='Path for annotated output video')
    vid_parser.add_argument('--start-frame', type=int, default=0)
    vid_parser.add_argument('--end-frame', type=int, default=None)
    vid_parser.add_argument('--window-size', type=int, default=41)
    vid_parser.add_argument('--stride', type=int, default=5,
                            help='Frames between each prediction window (default 5, lower = more readings)')
    vid_parser.add_argument('--frame-skip', type=int, default=2,
                            help='Process every Nth frame (default 2, higher = faster)')
    vid_parser.add_argument('--crop', type=str, default=None,
                            help='Crop region: "left", "right", "top", "bottom", '
                                 'or custom "L,T,R,B" as fractions (e.g. "0.0,0.0,0.5,1.0")')

    # ── live subcommand ──
    live_parser = subparsers.add_parser('live', help='Predict from live camera')
    live_parser.add_argument('--camera', type=int, default=0, help='Camera index')
    live_parser.add_argument('--model', default=None, help='Path to LSTM model')
    live_parser.add_argument('--window-size', type=int, default=41)

    args = parser.parse_args()

    if args.command == 'video':
        # Parse crop region
        crop_region = None
        if args.crop:
            crop_presets = {
                'left':   (0.0, 0.0, 0.5, 1.0),
                'right':  (0.5, 0.0, 1.0, 1.0),
                'top':    (0.0, 0.0, 1.0, 0.5),
                'bottom': (0.0, 0.5, 1.0, 1.0),
            }
            if args.crop.lower() in crop_presets:
                crop_region = crop_presets[args.crop.lower()]
            else:
                try:
                    parts = [float(x) for x in args.crop.split(',')]
                    if len(parts) != 4:
                        raise ValueError
                    crop_region = tuple(parts)
                except ValueError:
                    print(f"ERROR: Invalid --crop value: '{args.crop}'")
                    print("  Use 'left', 'right', 'top', 'bottom', or 'L,T,R,B' (e.g. '0.0,0.0,0.5,1.0')")
                    sys.exit(1)

        output_path = args.output
        if output_path is None:
            output_dir = Path(__file__).resolve().parent / "output"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = str(output_dir / "result.mp4")
        elif not os.path.isabs(output_path) and os.sep not in output_path:
            # If they just provided a filename like "my_video.mp4", put it in the output dir
            output_dir = Path(__file__).resolve().parent / "output"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = str(output_dir / output_path)

        predictor = ShotPredictor(model_path=args.model, model_type=args.model_type)
        results = predictor.predict_from_video(
            args.video_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_video=output_path,
            window_size=args.window_size,
            stride=args.stride,
            frame_skip=args.frame_skip,
            crop_region=crop_region,
        )

        # Filter: only show actual shot detections in the summary
        shot_results = [r for r in results if r.is_shot]
        no_shot_count = len(results) - len(shot_results)

        print("\n" + "═" * 60)
        print("  SHOT CLASSIFICATION RESULTS")
        print("═" * 60)
        if no_shot_count > 0:
            print(f"  ({no_shot_count} windows suppressed — no arm swing detected)")
            print()
        if not shot_results:
            print("  No shots detected in this video.")
            print("  The player may have been doing footwork or not hitting.")
        for i, pred in enumerate(shot_results):
            print(f"  [{i+1}] Frames {pred.frame_range[0]:>5}–{pred.frame_range[1]:<5}  "
                  f"{pred.shot_type.upper():<8} {pred.confidence:>6.1%}")
            for shot, prob in sorted(pred.all_probabilities.items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 30)
                print(f"        {shot:<8} {prob:5.1%} {bar}")
        print("═" * 60)

    elif args.command == 'live':
        predictor = ShotPredictor(model_path=args.model)
        predictor.predict_live(
            camera_index=args.camera,
            window_size=args.window_size,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
