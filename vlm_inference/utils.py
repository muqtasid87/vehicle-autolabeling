import re
import json5
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import uuid

def normalize_json5(s: str, img_name: str):
    s = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", s).strip()
    try:
        return json5.loads(s)
    except Exception as e:
        print(f"normalize_json5: failed to parse input as JSON5 for image: {img_name}.")
        print("Error message:", str(e))
        print("Faulty string content:")
        print(s)
        return None

def numpy_to_base64(image: np.ndarray) -> str:
    success, encoded_img = cv2.imencode('.jpg', image)
    if not success:
        return None
    base64_data = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"

def base64_to_pil(base64_str: str) -> Image.Image:
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding base64 string or opening image: {e}")
        return None

def create_image_json(image_path: Path, image: np.ndarray, detections: list) -> dict:
    image_results = {
        "image_filename": image_path.name,
        "image_path": str(image_path),
        "image_dimensions": {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2]
        },
        "detection_count": len(detections),
        "detections": detections,
    }
    return image_results