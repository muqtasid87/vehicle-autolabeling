# src/vlm_inspector/vehicle_detector.py

import torch
import cv2
import uuid
from pathlib import Path
from .utils import numpy_to_base64

class VehicleDetector:
    """Handles vehicle detection in images using a YOLOv5 model."""
    CLASS_NAMES = {
        1: 'class1_lightVehicle', 2: 'class2_mediumVehicle', 3: 'class3_heavyVehicle',
        4: 'class4_taxi', 5: 'class5_bus', 6: 'class_motocycle'
    }
    VEHICLE_CLASSES = [1, 2, 3, 4, 5, 6]

    def __init__(self, model_path: str, conf_threshold: float = 0.1):
        self.conf_threshold = conf_threshold
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
            self.model.conf = self.conf_threshold
            print("YOLOv5 model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv5 model from '{model_path}': {e}")
            raise

    def detect(self, image: 'np.ndarray'):
        """Detects vehicles, crops them, and returns detection data."""
        img_height, img_width = image.shape[:2]
        vehicle_detections = []

        results = self.model(image)
        detections = results.pandas().xyxy[0]

        for _, det in detections.iterrows():
            if int(det['class']) not in self.VEHICLE_CLASSES or float(det['confidence']) < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue
            
            base64_crop = numpy_to_base64(vehicle_crop)
            if not base64_crop:
                continue
            
            detection_info = {
                "detection_id": str(uuid.uuid4()),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "class": self.CLASS_NAMES.get(int(det['class'])),
                "confidence": float(det['confidence']),
                "crop_base64": base64_crop
            }
            vehicle_detections.append(detection_info)
            
        return vehicle_detections