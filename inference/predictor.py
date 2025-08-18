from unsloth import FastVisionModel
import os
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from peft import PeftModel
from .yolo_detector import YoloDetector
from .utils import base64_to_pil, clean_and_parse_json
from common.logging_setup import setup_logging_and_dir
import config

logger = logging.getLogger(__name__)

class VlmPredictor:
    def __init__(self, model_name, input_folder, lora_adapter_path, yolo_model_path, batch_size, use_lora=True):
        self.model_name = model_name.lower()
        self.input_folder = Path(input_folder)
        self.lora_adapter_path = lora_adapter_path
        self.batch_size = batch_size
        self.use_lora = use_lora

        model_type = "Base" if not self.use_lora else "Finetuned"
        self.output_dir = setup_logging_and_dir("Inference", f"{model_name.capitalize()}_{model_type}")
        self.json_output_dir = Path(self.output_dir) / "json_outputs"
        self.json_output_dir.mkdir(exist_ok=True)
        self.error_log_path = Path(self.output_dir) / "parsing_errors.log"

        self.yolo_detector = YoloDetector(yolo_model_path)
        self._load_vlm()

    def _load_vlm(self):
        """Loads the base VLM and optionally attaches the fine-tuned LoRA adapters."""
        logger.info(f"Loading base model for '{self.model_name}'...")
        if self.model_name == 'qwen':
            base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif self.model_name == 'gemma':
            base_model_id = "unsloth/gemma-3-4b-pt" # Using the same base as for finetuning
        else:
            raise ValueError("Unsupported model_name. Choose 'qwen' or 'gemma'.")

        # Load base model
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            base_model_id,
            load_in_4bit=True,
        )
        
        # Optionally merge with LoRA adapters
        if self.use_lora:
            if self.lora_adapter_path:
                logger.info("Base model loaded. Merging with LoRA adapters...")
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
                logger.info("LoRA adapters merged successfully.")
            else:
                logger.warning("use_lora=True but no lora_adapter_path provided. Using base model only.")
                self.use_lora = False
        else:
            logger.info("Using base model only (no LoRA adapters).")

    def _vlm_inference_batch(self, batch_of_crops):
        """Performs inference on a batch of image crops."""
        pil_images = [base64_to_pil(item['crop_base64']) for item in batch_of_crops]
        valid_pil_images = [img for img in pil_images if img is not None]

        if not valid_pil_images:
            return [{'error': 'Image conversion failed'} for _ in batch_of_crops]

        # Prepare batch conversations
        conversations = []
        for image in valid_pil_images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{config.SYSTEM_PROMPT}\n\n{config.USER_PROMPT}"},
                    {"type": "image", "image": image},
                ]
            }]
            conversations.append(messages)
        
        # Alternative approach: process conversations one by one if batch processing fails
        try:
            inputs = self.tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
            )
            
            # Move inputs to device and ensure it's in the correct format
            if isinstance(inputs, torch.Tensor):
                # If inputs is just a tensor, wrap it in a dictionary
                inputs = {"input_ids": inputs.to(self.model.device)}
            else:
                # If inputs is already a dict, move all tensors to device
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                decoded_outputs = self.tokenizer.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        except Exception as e:
            # Fallback: process one by one
            logger.warning(f"Batch processing failed: {e}. Processing images individually.")
            decoded_outputs = []
            for messages in conversations:
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        [messages],  # Single conversation
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                    )
                    
                    if isinstance(inputs, torch.Tensor):
                        inputs = {"input_ids": inputs.to(self.model.device)}
                    else:
                        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                    
                    with torch.inference_mode():
                        generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                        decoded_output = self.tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                        decoded_outputs.append(decoded_output)
                except Exception as inner_e:
                    logger.error(f"Failed to process individual image: {inner_e}")
                    decoded_outputs.append("")

        results = []
        img_idx = 0
        for i, pil_img in enumerate(pil_images):
            if pil_img is not None:
                parsed_json = clean_and_parse_json(
                    decoded_outputs[img_idx],
                    batch_of_crops[i]['img_name_for_vlm'],
                    self.error_log_path
                )
                results.append(parsed_json)
                img_idx += 1
            else:
                results.append({'error': 'Image conversion failed'})
        return results

    def run(self):
        """Main processing loop for inference."""
        start_time = time.perf_counter()
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [p for ext in image_exts for p in self.input_folder.glob(f'*{ext}')]
        logger.info(f"Found {len(image_files)} images in '{self.input_folder}'.")

        all_crops_to_process = []
        # Stage 1: Run YOLO on all images and collect crops
        logger.info("Stage 1: Running YOLO detection and collecting crops...")
        for img_path in tqdm(image_files, desc="YOLO Detection"):
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not read image: {img_path.name}")
                continue
            
            yolo_detections = self.yolo_detector.detect(image)
            for idx, detection in enumerate(yolo_detections):
                detection['original_image_path'] = str(img_path)
                detection['img_name_for_vlm'] = f"{img_path.name}_crop_{idx}"
                all_crops_to_process.append(detection)

        logger.info(f"Collected a total of {len(all_crops_to_process)} vehicle crops to analyze.")

        # Stage 2: Process crops in batches with VLM
        logger.info("Stage 2: Running VLM inference on cropped vehicles...")
        vlm_results = {} # Key: image_path, Value: list of detection results
        for i in tqdm(range(0, len(all_crops_to_process), self.batch_size), desc="VLM Inference"):
            batch = all_crops_to_process[i:i+self.batch_size]
            batch_vlm_outputs = self._vlm_inference_batch(batch)
            
            for detection_info, vlm_output in zip(batch, batch_vlm_outputs):
                original_path = detection_info['original_image_path']
                if original_path not in vlm_results:
                    vlm_results[original_path] = []
                
                # Update detection info with VLM output
                detection_info['vlm_inference'] = vlm_output
                detection_info.pop('crop_base64') # Remove base64 to save space in JSON
                vlm_results[original_path].append(detection_info)

        # Stage 3: Collate results and save one JSON per image
        logger.info("Stage 3: Saving JSON results for each image...")
        for img_path_str, detections in tqdm(vlm_results.items(), desc="Saving JSONs"):
            img_path = Path(img_path_str)
            image_cv = cv2.imread(img_path_str)
            height, width, channels = image_cv.shape

            final_json = {
                "image_filename": img_path.name,
                "image_path": img_path_str,
                "image_dimensions": {"width": width, "height": height, "channels": channels},
                "detection_count": len(detections),
                "detections": detections
            }
            
            output_json_path = self.json_output_dir / f"{img_path.stem}.json"
            with open(output_json_path, 'w') as f:
                json.dump(final_json, f, indent=4)

        end_time = time.perf_counter()
        logger.info(f"Processed {len(image_files)} images in {end_time - start_time:.2f} seconds.")
        return self.output_dir