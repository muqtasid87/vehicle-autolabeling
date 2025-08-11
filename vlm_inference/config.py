import os

# Model configs
QWEN_BASE_MODEL_ID = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
QWEN_LORA_REPO = "muqtasid87/qwen2.5-filtered-data-qv-final-push"
GEMMA_BASE_MODEL_ID = "unsloth/gemma-3-4b-it-bnb-4bit"
GEMMA_LORA_REPO = "muqtasid87/gemma_fyp_lora_adapters"
YOLO_MODEL_PATH = "best_avc_v5.pt"
YOLO_MODEL_GDRIVE_ID = "1SrbA34-ptAA7Nm92TpCcHMn3lvwh-EKh"

# Prompts (replace placeholders with actual content)
SYSTEM_PROMPT_COT = """
<system prompt with Chain of Thought Placeholder>
"""
SYSTEM_PROMPT_NO_COT = """
<system prompt without Chain of Thought Placeholder>
"""
USER_PROMPT_COT = """
<user prompt with Chain of Thought Placeholder>

**Output JSON keys & descriptors:**
1. **CoT**: Your step-by-step observations...
# (full as in your Qwen code)
"""
USER_PROMPT_NO_COT = """
<user prompt without Chain of Thought Placeholder>

**Output JSON keys & descriptors:**
1. **Category**: The main vehicle type...
# (full as in your Qwen code)
"""

def get_prompts(cot: bool):
    system_prompt = SYSTEM_PROMPT_COT if cot else SYSTEM_PROMPT_NO_COT
    user_prompt = USER_PROMPT_COT if cot else USER_PROMPT_NO_COT
    return system_prompt, user_prompt