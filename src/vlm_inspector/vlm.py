# src/vlm_inspector/vlm.py

import torch
from abc import ABC, abstractmethod
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration
from peft import PeftModel
from .utils import normalize_json5, base64_to_pil

class VLMInference(ABC):
    """Abstract base class for VLM inference models."""
    def __init__(self, use_finetuned: bool, system_prompt: str, user_prompt: str):
        self.model = None
        self.processor = None
        self.use_finetuned = use_finetuned
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load_model(self):
        """Loads the model and processor."""
        pass

    @abstractmethod
    def infer_batch(self, imgs_base64_list: list, img_names_list: list):
        """Performs inference on a batch of images."""
        pass

class QwenVLM(VLMInference):
    """Qwen-VL-specific inference implementation."""
    BASE_MODEL_ID = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
    LORA_REPO = "muqtasid87/qwen2.5-filtered-data-qv-final-push"

    def load_model(self):
        print("Loading Qwen model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.BASE_MODEL_ID,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL_ID)
        
        if self.use_finetuned:
            print(f"Loading LoRA adapters from {self.LORA_REPO}...")
            self.model = PeftModel.from_pretrained(
                self.model, self.LORA_REPO, is_trainable=False
            ).to(self.device)
        print("Qwen model loaded successfully.")

    def infer_batch(self, imgs_base64_list: list, img_names_list: list):
        pil_images = [base64_to_pil(b64) for b64 in imgs_base64_list]
        valid_indices = [i for i, img in enumerate(pil_images) if img is not None]
        
        if not valid_indices:
            return [normalize_json5('{"error": "Image conversion failed."}', name) for name in img_names_list]

        messages_for_template = [
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.user_prompt}]}]
            for _ in valid_indices
        ]
        
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_for_template]
        valid_pil_images = [pil_images[i] for i in valid_indices]

        inputs = self.processor(
            text=texts, images=valid_pil_images, padding=True, return_tensors="pt"
        ).to(self.device, dtype=torch.float16)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=False)
        trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded_outputs = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)

        final_results = [None] * len(imgs_base64_list)
        valid_output_idx = 0
        for i in range(len(imgs_base64_list)):
            if i in valid_indices:
                final_results[i] = normalize_json5(decoded_outputs[valid_output_idx], img_names_list[i])
                valid_output_idx += 1
            else:
                final_results[i] = normalize_json5('{"error": "Image conversion failed."}', img_names_list[i])
        
        return final_results

class GemmaVLM(VLMInference):
    """Gemma-specific inference implementation."""
    BASE_MODEL_ID = "unsloth/gemma-3-4b-it-bnb-4bit"
    LORA_REPO = "muqtasid87/gemma_fyp_lora_adapters"

    def load_model(self):
        print("Loading Gemma model...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.BASE_MODEL_ID,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL_ID)

        if self.use_finetuned:
            print(f"Loading LoRA adapters from {self.LORA_REPO}...")
            self.model = PeftModel.from_pretrained(
                self.model, self.LORA_REPO, is_trainable=False
            ).to(self.device)
        print("Gemma model loaded successfully.")

    def infer_batch(self, imgs_base64_list: list, img_names_list: list):
        batch_conversations = []
        for i, img_b64 in enumerate(imgs_base64_list):
            # Gemma's template is slightly different
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.user_prompt}]}
            ]
            batch_conversations.append(messages)

        # Note: Gemma's processor can take images directly in apply_chat_template
        pil_images = [base64_to_pil(b64) for b64 in imgs_base64_list]
        
        inputs = self.processor.apply_chat_template(
            batch_conversations,
            images=pil_images, # Pass PIL images here
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=False)
        
        decoded_outputs = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return [normalize_json5(output, name) for output, name in zip(decoded_outputs, img_names_list)]