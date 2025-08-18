import base64
import json
import logging
import re
from io import BytesIO
import json5
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

def numpy_to_base64(image):
    """Converts a numpy array image to a base64 encoded string."""
    success, encoded_img = cv2.imencode('.jpg', image)
    if not success:
        return None
    base64_data = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"


def base64_to_pil(base64_str):
    """Converts a base64 string to a PIL Image object."""
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"Error decoding base64 string: {e}")
        return None

def clean_and_parse_json(vlm_output: str, image_name: str, error_log_path: str):
    """
    Cleans and parses a JSON string from VLM output with robust error handling.

    Args:
        vlm_output (str): The raw string output from the VLM.
        image_name (str): The name of the image for logging purposes.
        error_log_path (str): Path to the log file for parsing errors.

    Returns:
        dict: A parsed Python dictionary, or an error dictionary on failure.
    """
    # 1. Remove markdown fences (e.g., ```json ... ```)
    cleaned_str = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", vlm_output).strip()
    
    # 2. Handle escaped quotes that might be misplaced by the model
    cleaned_str = cleaned_str.replace(r'\"', '"')
    
    try:
        # 3. Use json5 for lenient parsing (handles trailing commas, etc.)
        return json5.loads(cleaned_str)
    except Exception as e:
        error_message = (
            f"Failed to parse JSON for image '{image_name}'. Error: {e}\n"
            f"--- Original VLM Output ---\n{vlm_output}\n"
            f"--- Cleaned String Attempted ---\n{cleaned_str}\n"
            "----------------------------\n"
        )
        logger.error(f"JSON parsing failed for {image_name}. See {error_log_path} for details.")
        with open(error_log_path, 'a') as f:
            f.write(error_message)
        
        return {"error": "JSON parsing failed", "details": str(e)}