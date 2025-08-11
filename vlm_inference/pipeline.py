import time
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from .detectors import YOLODetector
from .utils import create_image_json
from .config import get_prompts

def process_images(
    input_folder: str,
    output_folder: str,
    vlm_instance,  # Instance of QwenVLM or GemmaVLM
    conf_threshold: float = 0.1,
    vlm_batch_size: int = 2
):
    yolo_detector = YOLODetector(conf_threshold)
    os.makedirs(output_folder, exist_ok=True)
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_input_files = [p for ext in image_exts for p in Path(input_folder).glob(f'*{ext}')]
    print(f"Found {len(all_input_files)} images in '{input_folder}'")
    start_all_time = time.perf_counter()

    all_images_data = {}
    crops_to_vlm_batch = []
    processed_image_count = 0
    total_yolo_detections_count = 0
    total_vlm_inferences_done = 0

    for img_path in tqdm(all_input_files, desc="YOLO & Crop Collection", unit="img"):
        img_path_str = str(img_path)
        img_name = img_path.name
        img = cv2.imread(img_path_str)
        if img is None:
            tqdm.write(f"[SKIP] Could not read '{img_name}'")
            all_images_data[img_path_str] = {'img_cv2': None, 'yolo_detections_with_vlm': []}
            continue
        all_images_data[img_path_str] = {'img_cv2': img, 'yolo_detections_with_vlm': []}
        yolo_detections = yolo_detector.detect_vehicles(img)
        total_yolo_detections_count += len(yolo_detections)
        for det_idx, yolo_det_obj in enumerate(yolo_detections):
            crop_b64 = yolo_det_obj.pop("crop_base64", None)
            if crop_b64:
                crops_to_vlm_batch.append({
                    'crop_b64': crop_b64,
                    'img_name_for_vlm': f"{img_name}_crop_{det_idx}",
                    'image_path_str': img_path_str,
                    'original_det_idx': det_idx,
                    'yolo_det_obj': yolo_det_obj
                })
        if len(crops_to_vlm_batch) >= vlm_batch_size:
            tqdm.write(f"Processing VLM batch of {len(crops_to_vlm_batch)} crops...")
            b64_list = [item['crop_b64'] for item in crops_to_vlm_batch]
            img_names = [item['img_name_for_vlm'] for item in crops_to_vlm_batch]
            vlm_results = vlm_instance.fine_grained_inference_batched(b64_list, img_names)
            total_vlm_inferences_done += len(vlm_results)
            for i, vlm_output in enumerate(vlm_results):
                item_info = crops_to_vlm_batch[i]
                item_info['yolo_det_obj']['vlm_fine_grained_inference'] = vlm_output
                all_images_data[item_info['image_path_str']]['yolo_detections_with_vlm'].append(item_info['yolo_det_obj'])
            crops_to_vlm_batch = []

    if crops_to_vlm_batch:
        tqdm.write(f"Processing final VLM batch of {len(crops_to_vlm_batch)} crops...")
        b64_list = [item['crop_b64'] for item in crops_to_vlm_batch]
        img_names = [item['img_name_for_vlm'] for item in crops_to_vlm_batch]
        vlm_results = vlm_instance.fine_grained_inference_batched(b64_list, img_names)
        total_vlm_inferences_done += len(vlm_results)
        for i, vlm_output in enumerate(vlm_results):
            item_info = crops_to_vlm_batch[i]
            item_info['yolo_det_obj']['vlm_fine_grained_inference'] = vlm_output
            all_images_data[item_info['image_path_str']]['yolo_detections_with_vlm'].append(item_info['yolo_det_obj'])

    for img_path_str, data in tqdm(all_images_data.items(), desc="Creating JSON files", unit="file"):
        img_p = Path(img_path_str)
        image_cv2_obj = data['img_cv2']
        if image_cv2_obj is None:
            img_json_data = create_image_json(img_p, np.zeros((100, 100, 3), dtype=np.uint8), [])
        else:
            img_json_data = create_image_json(img_p, image_cv2_obj, data['yolo_detections_with_vlm'])
        out_file = Path(output_folder) / f"{img_p.stem}_detections.json"
        with open(out_file, 'w') as f:
            json.dump(img_json_data, f, indent=2)
        processed_image_count += 1

    end_all_time = time.perf_counter()
    total_duration = end_all_time - start_all_time
    avg_time = total_duration / len(all_input_files) if all_input_files else 0

    print(f"\n--- Processing Summary ---")
    print(f"Total images processed: {processed_image_count} / {len(all_input_files)}")
    print(f"Total YOLO detections found: {total_yolo_detections_count}")
    print(f"Total VLM inferences performed: {total_vlm_inferences_done}")
    print(f"Total pipeline time: {total_duration:.2f} seconds")
    print(f"Average time per image: {avg_time:.2f} seconds")