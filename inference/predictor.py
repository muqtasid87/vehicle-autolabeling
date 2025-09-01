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
from config import USER_PROMPT, SYSTEM_PROMPT
import pathlib
import platform

# Only patch if running on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

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
            base_model_id = "unsloth/gemma-3-4b-it" 
        else:
            raise ValueError("Unsupported model_name. Choose 'qwen' or 'gemma'.")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            base_model_id,
            load_in_4bit=True,
        )

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

    # --------------------------------------------------------------------------
    # --- MODEL-SPECIFIC BATCHING METHODS ---
    # --------------------------------------------------------------------------

    def _vlm_inference_batch_qwen(self, batch_of_crops):
        """
        Optimized batch processing for Qwen, which uses a standard transformers API.
        """
        if not batch_of_crops: return []
        pil_images = [base64_to_pil(item['crop_base64']) for item in batch_of_crops]

        valid_pil_images, valid_indices = [], []
        for i, img in enumerate(pil_images):
            if img is not None:
                valid_pil_images.append(img)
                valid_indices.append(i)

        if not valid_pil_images:
            return [{'error': 'All image conversions failed in batch'}] * len(batch_of_crops)

        try:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
            text_prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(
                text=[text_prompts] * len(valid_pil_images),
                images=valid_pil_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

            input_ids_len = inputs["input_ids"].shape[1]
            decoded_outputs = self.tokenizer.batch_decode(generated_ids[:, input_ids_len:], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Qwen batch processing failed: {e}", exc_info=True)
            decoded_outputs = [f"Error: {e}"] * len(valid_pil_images)

        # Collate results
        final_results = [None] * len(batch_of_crops)
        for i, idx in enumerate(valid_indices):
            final_results[idx] = decoded_outputs[i]
        for i in range(len(final_results)):
            if final_results[i] is None:
                final_results[i] = {'error': 'Image conversion failed'}
            else:
                final_results[i] = clean_and_parse_json(
                    final_results[i], batch_of_crops[i]['img_name_for_vlm'], self.error_log_path
                )
        return final_results


    def _vlm_inference_batch_gemma(self, batch_of_crops):
        """
        Manual batch processing for Gemma/Unsloth, which has a non-standard processor API.
        """
        if not batch_of_crops: return []
        pil_images = [base64_to_pil(item['crop_base64']) for item in batch_of_crops]

        all_inputs, valid_indices = [], []
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        for i, img in enumerate(pil_images):
            if img is not None:
                valid_indices.append(i)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}, {"type": "image"}]}
                ]
                text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(text=text_prompt, images=img, return_tensors="pt")
                all_inputs.append(inputs)

        if not all_inputs:
            return [{'error': 'All image conversions failed'}] * len(batch_of_crops)

        try:
            max_len = max(inp['input_ids'].shape[1] for inp in all_inputs)
            batch_size = len(all_inputs)
            padded_input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

            for i, inp in enumerate(all_inputs):
                seq_len = inp['input_ids'].shape[1]
                padded_input_ids[i, :seq_len] = inp['input_ids'][0]
                padded_attention_mask[i, :seq_len] = 1

            batched_pixels = torch.cat([inp['pixel_values'] for inp in all_inputs], dim=0)

            batched_inputs = {
                "input_ids": padded_input_ids, "attention_mask": padded_attention_mask, "pixel_values": batched_pixels
            }
            batched_inputs = {key: value.to(self.model.device) for key, value in batched_inputs.items()}

            with torch.inference_mode():
                generated_ids = self.model.generate(**batched_inputs, max_new_tokens=512, do_sample=False)

            input_ids_len = batched_inputs["input_ids"].shape[1]
            decoded_outputs = self.tokenizer.batch_decode(generated_ids[:, input_ids_len:], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Gemma batch processing failed: {e}", exc_info=True)
            decoded_outputs = [f"Error: {e}"] * len(all_inputs)

        # Collate results
        final_results = [None] * len(batch_of_crops)
        for i, idx in enumerate(valid_indices):
            final_results[idx] = decoded_outputs[i]
        for i in range(len(final_results)):
            if final_results[i] is None:
                final_results[i] = {'error': 'Image conversion failed'}
            else:
                final_results[i] = clean_and_parse_json(
                    final_results[i], batch_of_crops[i]['img_name_for_vlm'], self.error_log_path
                )
        return final_results

    # --------------------------------------------------------------------------
    # --- SAVE METHOD AND RUN METHOD ---
    # --------------------------------------------------------------------------

    def _save_result_json(self, image_path_str, detections):
        """Saves the final JSON output for a single image."""
        img_path = Path(image_path_str)
        try:
            image_cv = cv2.imread(image_path_str)
            if image_cv is None:
                raise IOError(f"Could not read image file to get dimensions: {img_path.name}")
            height, width, channels = image_cv.shape

            final_json = {
                "image_filename": img_path.name,
                "image_path": image_path_str,
                "image_dimensions": {"width": width, "height": height, "channels": channels},
                "detection_count": len(detections),
                "detections": detections
            }

            output_json_path = self.json_output_dir / f"{img_path.stem}.json"
            with open(output_json_path, 'w') as f:
                json.dump(final_json, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save JSON for {img_path.name}: {e}", exc_info=True)


    def run(self):
        """Main processing loop for inference with immediate JSON saving."""
        start_time = time.perf_counter()
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [p for ext in image_exts for p in self.input_folder.glob(f'*{ext}')]
        logger.info(f"Found {len(image_files)} images in '{self.input_folder}'.")

        # Stage 1: Collect all crops and count expected detections per image
        all_crops_to_process = []
        image_crop_counts = defaultdict(int)
        for img_path in tqdm(image_files, desc="YOLO Detection"):
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not read image: {img_path.name}")
                continue

            yolo_detections = self.yolo_detector.detect(image)
            image_crop_counts[str(img_path)] = len(yolo_detections) # Track total crops per image

            for idx, detection in enumerate(yolo_detections):
                detection['original_image_path'] = str(img_path)
                detection['img_name_for_vlm'] = f"{img_path.name}_crop_{idx}"
                all_crops_to_process.append(detection)

        logger.info(f"Collected a total of {len(all_crops_to_process)} vehicle crops to analyze.")

        # Stage 2: Run VLM inference and save JSONs as soon as an image is complete
        vlm_results_aggregator = defaultdict(list)
        processed_image_paths = set()

        logger.info("Running VLM inference and saving results...")
        for i in tqdm(range(0, len(all_crops_to_process), self.batch_size), desc="VLM Inference & Saving"):
            batch = all_crops_to_process[i:i+self.batch_size]

            if self.model_name == 'qwen':
                batch_vlm_outputs = self._vlm_inference_batch_qwen(batch)
            elif self.model_name == 'gemma':
                batch_vlm_outputs = self._vlm_inference_batch_gemma(batch)
            else:
                raise NotImplementedError(f"No batch processing method implemented for model: {self.model_name}")

            # Aggregate results from the batch
            for detection_info, vlm_output in zip(batch, batch_vlm_outputs):
                original_path = detection_info['original_image_path']
                detection_info['vlm_inference'] = vlm_output
                detection_info.pop('crop_base64')
                vlm_results_aggregator[original_path].append(detection_info)

            # Check if any images in the batch are now complete and save them
            paths_in_batch = {d['original_image_path'] for d in batch}
            for path in paths_in_batch:
                if path not in processed_image_paths:
                    if len(vlm_results_aggregator[path]) == image_crop_counts[path]:
                        logger.info(f"All crops for {Path(path).name} processed. Saving JSON.")
                        self._save_result_json(path, vlm_results_aggregator[path])
                        processed_image_paths.add(path)
                        del vlm_results_aggregator[path] # Free up memory

        # Final check for any images that had zero detections (to create empty JSONs)
        logger.info("Creating empty JSON files for images with no detections...")
        for img_path in image_files:
            path_str = str(img_path)
            if image_crop_counts[path_str] == 0:
                 self._save_result_json(path_str, [])

        end_time = time.perf_counter()
        logger.info(f"Processed {len(image_files)} images in {end_time - start_time:.2f} seconds.")
        logger.info(f"Saved {len(processed_image_paths)} JSON files with detections.")
        return self.output_dir