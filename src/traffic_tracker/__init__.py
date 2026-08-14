"""
Traffic Tracker AI
==================
An end-to-end computer vision system for real-time:
  - Vehicle detection & multi-object tracking
  - License plate detection & OCR
  - Vehicle color classification
  - Vehicle body-type classification

Usage:
    from traffic_tracker.pipeline import TrafficPipeline
    pipeline = TrafficPipeline()
    results = pipeline.process_frame(frame)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
__author__ = "Traffic Tracker AI"

