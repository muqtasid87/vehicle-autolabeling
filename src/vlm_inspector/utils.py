# src/vlm_inspector/utils.py

import re
import json5
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

def normalize_json5(s: str, img_name: str):
    """Strips markdown fences and parses using json5 for lenient JSON parsing."""
    s = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", s).strip()
    try:
        return json5.loads(s)
    except Exception as e:
        print(f"normalize_json5: failed to parse input as JSON5 for image: {img_name}.")
        print(f"Error message: {str(e)}")
        print("Faulty string content:\n", s)
        return {"error": "JSON parsing failed", "details": str(e), "original_text": s}

def numpy_to_base64(image: np.ndarray) -> str:
    """Converts a numpy array image to a base64 encoded string."""
    success, encoded_img = cv2.imencode('.jpg', image)
    if not success:
        return None
    base64_data = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"

def base64_to_pil(base64_str: str) -> Image.Image:
    """Converts a base64 string to a PIL Image."""
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"Error decoding base64 string or opening image: {e}")
        return None