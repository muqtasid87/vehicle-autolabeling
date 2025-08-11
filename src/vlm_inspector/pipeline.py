# src/vlm_inspector/pipeline.py

import os
import json
import cv2
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .vehicle_detector import VehicleDetector
from .vlm import QwenVLM, GemmaVLM
from .prompts import get_prompts

class InferencePipeline:
    def __init__(self, config: dict):
        print("Initializing Inference Pipeline...")
        self.config = config
        self.detector = VehicleDetector(
            model_path=config['yolo_model_path'],
            conf_threshold=config['conf_threshold']
        )
        
        system_prompt, user_prompt = get_prompts(cot=config['use_cot'])
        
        if config['vlm_model'] == 'qwen':
            self.vlm = QwenVLM(config['use_finetuned'], system_prompt, user_prompt)
        elif config['vlm_model'] == 'gemma':
            self.vlm = GemmaVLM(config['use_finetuned'], system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported VLM model: {config['vlm_model']}")
            
        self.vlm.load_model()
        print("Pipeline initialized.")

    def run(self, input_folder: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        all_files = [p for ext in image_exts for p in Path(input_folder).glob(f'**/*{ext}')]
        
        print(f"Found {len(all_files)} images in '{input_folder}'.")
        start_time = time.perf_counter()

        all_crops_to_process = []
        # Stage 1: YOLO Detection and Crop Collection
        print("Stage 1: Running vehicle detection...")
        for img_path in tqdm(all_files, desc="Detecting Vehicles"):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            detections = self.detector.detect(image)
            for i, det in enumerate(detections):
                det['source_image_path'] = str(img_path)
                det['crop_name'] = f"{img_path.name}_crop_{i}"
                all_crops_to_process.append(det)

        # Stage 2: Batched VLM Inference
        print(f"\nStage 2: Running VLM inference on {len(all_crops_to_process)} detected vehicles...")
        vlm_results = {}
        batch_size = self.config['vlm_batch_size']
        for i in tqdm(range(0, len(all_crops_to_process), batch_size), desc="VLM Inference Batches"):
            batch = all_crops_to_process[i:i + batch_size]
            b64_list = [item['crop_base64'] for item in batch]
            name_list = [item['crop_name'] for item in batch]
            
            inference_results = self.vlm.infer_batch(b64_list, name_list)
            
            for item, result in zip(batch, inference_results):
                # We no longer need the large base64 string
                del item['crop_base64']
                item['vlm_inference'] = result
                
                # Group results by original image path
                source_path = item['source_image_path']
                if source_path not in vlm_results:
                    vlm_results[source_path] = []
                vlm_results[source_path].append(item)

        # Stage 3: Writing JSON Outputs
        print("\nStage 3: Writing output JSON files...")
        for img_path_str, detections in tqdm(vlm_results.items(), desc="Saving JSON files"):
            img_path = Path(img_path_str)
            image_shape = cv2.imread(img_path_str).shape
            
            output_data = {
                "image_filename": img_path.name,
                "image_path": img_path_str,
                "image_dimensions": {"height": image_shape[0], "width": image_shape[1]},
                "detection_count": len(detections),
                "detections": detections,
            }
            
            out_file = Path(output_folder) / f"{img_path.stem}_detections.json"
            with open(out_file, 'w') as f:
                json.dump(output_data, f, indent=2)

        duration = time.perf_counter() - start_time
        print(f"\n✅ Processing complete in {duration:.2f} seconds.")