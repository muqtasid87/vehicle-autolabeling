import logging
import uuid
import torch
from .utils import numpy_to_base64

logger = logging.getLogger(__name__)

class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.1):
        self.conf_threshold = conf_threshold
        self.class_names = {
            1: 'class1_lightVehicle', 2: 'class2_mediumVehicle', 3: 'class3_heavyVehicle',
            4: 'class4_taxi', 5: 'class5_bus', 6: 'class_motocycle'
        }
        self.vehicle_classes = list(self.class_names.keys())
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            model.conf = self.conf_threshold
            logger.info("YOLOv5 model loaded successfully using torch.hub.")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model from {model_path}: {e}")
            raise

    def detect(self, image):
        img_height, img_width = image.shape[:2]
        detections_list = []

        try:
            results = self.model(image)
            detections = results.pandas().xyxy[0]

            for _, det in detections.iterrows():
                class_id = int(det['class'])
                confidence = float(det['confidence'])

                if class_id not in self.vehicle_classes:
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
                    "class": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": confidence,
                    "crop_base64": base64_crop
                }
                detections_list.append(detection_info)

        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")

        return detections_list