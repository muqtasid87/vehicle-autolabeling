import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor as QwenProcessor
from transformers import PaliGemmaForConditionalGeneration as GemmaForConditionalGeneration, AutoProcessor as GemmaProcessor  # Assuming PaliGemma-like; adjust if needed
from peft import PeftModel
from .config import QWEN_BASE_MODEL_ID, QWEN_LORA_REPO, GEMMA_BASE_MODEL_ID, GEMMA_LORA_REPO
from .utils import base64_to_pil, normalize_json5

class VLMBase:
    def __init__(self, use_finetuned: bool, system_prompt: str, user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.use_finetuned = use_finetuned
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        raise NotImplementedError("Subclasses must implement model loading.")

    def fine_grained_inference_batched(self, imgs_base64_list: list, img_names_list: list = None) -> list:
        raise NotImplementedError("Subclasses must implement batched inference.")

class QwenVLM(VLMBase):
    def _load_model(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_BASE_MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = QwenProcessor.from_pretrained(QWEN_BASE_MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels)
        if self.use_finetuned:
            self.model = PeftModel.from_pretrained(self.model, QWEN_LORA_REPO, is_trainable=False).to("cuda")

    def fine_grained_inference_batched(self, imgs_base64_list: list, img_names_list: list = None) -> list:
        batch_size = len(imgs_base64_list)
        if batch_size == 0:
            return []
        if img_names_list is None:
            img_names_list = [f"image_in_batch_{i}" for i in range(batch_size)]
        batched_messages_for_template = []
        pil_images_for_processor = []
        for i, img_b64 in enumerate(imgs_base64_list):
            pil_image = base64_to_pil(img_b64)
            current_img_name = img_names_list[i]
            if pil_image is None:
                print(f"Warning: Failed to convert base64 image to PIL for {current_img_name}.")
                current_item_messages = [
                    {"role": "system", "content": f"Error: Image {current_img_name} could not be processed."},
                    {"role": "user", "content": [{"type": "text", "text": f"Error in image {current_img_name}."}]}
                ]
                pil_images_for_processor.append(None)
            else:
                pil_images_for_processor.append(pil_image)
                current_item_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.user_prompt}]}
                ]
            text_for_item = self.processor.apply_chat_template(current_item_messages, tokenize=False, add_generation_prompt=True)
            batched_messages_for_template.append(text_for_item)
        valid_indices = [i for i, img in enumerate(pil_images_for_processor) if img is not None]
        if not valid_indices:
            error_msg = '{"error": "Image conversion failed for all items in batch."}'
            return [normalize_json5(error_msg, img_names_list[0])] * batch_size
        final_pil_images = [pil_images_for_processor[i] for i in valid_indices]
        final_texts = [batched_messages_for_template[i] for i in valid_indices]
        inputs = self.processor(text=final_texts, images=final_pil_images, padding=True, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=False)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded_outputs = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        final_results = [None] * batch_size
        valid_idx = 0
        for original_idx in range(batch_size):
            if original_idx in valid_indices:
                final_results[original_idx] = normalize_json5(decoded_outputs[valid_idx], img_names_list[original_idx])
                valid_idx += 1
            else:
                final_results[original_idx] = normalize_json5('{"error": "Image conversion failed."}', img_names_list[original_idx])
        return final_results

class GemmaVLM(VLMBase):
    def _load_model(self):
        self.model = GemmaForConditionalGeneration.from_pretrained(
            GEMMA_BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
        self.processor = GemmaProcessor.from_pretrained(GEMMA_BASE_MODEL_ID)
        if self.use_finetuned:
            self.model = PeftModel.from_pretrained(self.model, GEMMA_LORA_REPO, is_trainable=False, torch_dtype=torch.bfloat16).to('cuda')

    def fine_grained_inference_batched(self, imgs_base64_list: list, img_names_list: list = None) -> list:
        batch_size = len(imgs_base64_list)
        if batch_size == 0:
            return []
        if img_names_list is None:
            img_names_list = [f"image_in_batch_{i}" for i in range(batch_size)]
        batch_of_conversations = []
        for i, img_b64 in enumerate(imgs_base64_list):
            pil_image = base64_to_pil(img_b64)  # Note: Your original code passes b64 directly; assuming processor handles, but using PIL for consistency
            current_img_name = img_names_list[i]
            if pil_image is None:
                print(f"Warning: Failed to convert base64 image to PIL for {current_img_name}.")
                current_item_messages = [
                    {"role": "system", "content": f"Error: Image {current_img_name} could not be processed."},
                    {"role": "user", "content": [{"type": "text", "text": f"Error in image {current_img_name}."}]}
                ]
            else:
                current_item_messages = [
                    {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                    {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": self.user_prompt}]}  # Adjusted to PIL
                ]
            batch_of_conversations.append(current_item_messages)
        inputs = self.processor.apply_chat_template(
            batch_of_conversations, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            generations = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        decoded_batch_results = []
        input_len = inputs["input_ids"].shape[-1]
        for i in range(generations.shape[0]):
            generation_item_tokens = generations[i][input_len:]
            decoded_text = self.processor.decode(generation_item_tokens, skip_special_tokens=True)
            decoded_batch_results.append(normalize_json5(decoded_text, img_names_list[i]))
        return decoded_batch_results