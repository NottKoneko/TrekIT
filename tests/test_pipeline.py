"""
test_pipeline.py
----------------
End-to-end unit tests for TrafficPipeline, AsyncTrafficPipeline, and multi-head VehicleClassifier.
"""

import numpy as np
from traffic_tracker.classifier import VehicleClassifier
from traffic_tracker.pipeline import AsyncTrafficPipeline, TrafficPipeline, load_config


def test_vehicle_classifier_batch():
    cfg = load_config("config.yaml")
    clf = VehicleClassifier(cfg)
    
    # Test batch predictions
    crops = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.full((120, 80, 3), 200, dtype=np.uint8),
    ]
    c_probs, t_probs, o_probs = clf.predict_attributes_batch(crops)
    assert c_probs.shape[0] == 2
    assert t_probs.shape[0] == 2
    assert o_probs.shape[0] == 2
    assert len(clf.orientation_classes) == 3


def test_traffic_pipeline_sync():
    pipeline = TrafficPipeline()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Process image
    ann_img, recs_img = pipeline.process_image(frame)
    assert ann_img.shape == (720, 1280, 3)
    assert isinstance(recs_img, list)

    # Process frame
    ann_f, recs_f = pipeline.process_frame(frame)
    assert ann_f.shape == (720, 1280, 3)
    assert isinstance(recs_f, list)
