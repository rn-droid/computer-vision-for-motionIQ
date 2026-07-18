# ABOUTME: Computer Vision Shot Prediction module for MotionIQ
# ABOUTME: Provides YOLO-based pose extraction and LSTM shot classification
"""
CV Model - Computer Vision Shot Prediction
==========================================

This package provides an end-to-end pipeline for predicting badminton shot
types from video using:
  1. YOLO pose estimation to extract 13 body keypoints
  2. Pre-trained LSTM model to classify shots into 6 categories:
     clear, drive, drop, lob, net, smash

Usage:
    from cv_model.inference import ShotPredictor

    predictor = ShotPredictor()
    results = predictor.predict_from_video("match_clip.mp4")
    print(results)
"""

__version__ = "0.1.0"
