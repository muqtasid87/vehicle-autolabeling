import json
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

def load_and_process_data(images_folder: str, json_folder: str):
    """
    Loads JSON annotations, crops images based on bounding boxes,
    and creates training samples.

    Args:
        images_folder (str): Path to the folder containing images.
        json_folder (str): Path to the folder containing JSON annotations.

    Returns:
        list: A list of training samples, each a dict with 'image' and 'ground_truth'.
    """
    training_samples = []
    json_files = list(Path(json_folder).glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {json_folder}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                annotation = json.load(f)

            image_filename = annotation["image"]["filename"]
            image_path = Path(images_folder) / image_filename

            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                continue

            original_image = Image.open(image_path).convert("RGB")

            for obj in annotation.get("objects", []):
                try:
                    bbox = obj["bbox"]  # [x1, y1, x2, y2]
                    cropped_image = original_image.crop(bbox)

                    ground_truth = {
                        "vehicle_info": obj.get("vehicle_info", {}),
                        "mechanical": obj.get("mechanical", {}),
                        "attributes": obj.get("attributes", {})
                    }

                    sample = {
                        "image": cropped_image,
                        "ground_truth": json.dumps(ground_truth, indent=2)
                    }
                    training_samples.append(sample)

                except Exception as e:
                    logger.error(f"Error processing object in {json_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue

    logger.info(f"Successfully created {len(training_samples)} training samples.")
    return training_samples