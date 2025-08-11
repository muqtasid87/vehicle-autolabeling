import torch
import os
import gdown
from .config import YOLO_MODEL_PATH, YOLO_MODEL_GDRIVE_ID
from .utils import numpy_to_base64

class YOLODetector:
    def __init__(self, conf_threshold: float = 0.1):
        self.conf_threshold = conf_threshold
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"Downloading YOLO model from Google Drive...")
            gdown.download(id=YOLO_MODEL_GDRIVE_ID, output=YOLO_MODEL_PATH, quiet=False)
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)
            model.conf = self.conf_threshold
        except Exception as e:
            try:
                model = torch.load(YOLO_MODEL_PATH, map_location='cpu')
                if hasattr(model, 'conf'):
                    model.conf = self.conf_threshold
            except Exception as e2:
                raise RuntimeError(f"Failed to load YOLO model: {e} / {e2}")
        return model

    def detect_vehicles(self, image: np.ndarray) -> list:
        class_names = {
            1: 'class1_lightVehicle',
            2: 'class2_mediumVehicle',
            3: 'class3_heavyVehicle',
            4: 'class4_taxi',
            5: 'class5_bus',
            6: 'class_motocycle'
        }
        vehicle_classes = [1, 2, 3, 4, 5, 6]
        img_height, img_width = image.shape[:2]
        vehicle_detections = []

        try:
            results = self.model(image)
            detections = results.pandas().xyxy[0]
            for _, detection in detections.iterrows():
                class_id = int(detection['class'])
                confidence = float(detection['confidence'])
                if class_id not in vehicle_classes or confidence < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                class_name = class_names.get(class_id, f"class_{class_id}")
                detection_id = str(uuid.uuid4())
                vehicle_crop = image[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue
                base64_crop = numpy_to_base64(vehicle_crop)
                detection_info = {
                    "detection_id": detection_id,
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1},
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                }
                if base64_crop:
                    detection_info["crop_base64"] = base64_crop  # Temporarily keep for VLM
                vehicle_detections.append(detection_info)
        except Exception as e:
            print(f"Error during YOLO inference: {str(e)}")
        return vehicle_detections