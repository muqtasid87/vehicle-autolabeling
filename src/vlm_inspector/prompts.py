# src/vlm_inspector/prompts.py

def get_prompts(cot: bool):
    """Returns the system and user prompts based on the Chain of Thought (CoT) flag."""
    if cot:
        system_prompt = "You are a vehicle expert" # TODO: Add system prompt, this is just a placeholder for testing
        user_prompt = """
<user prompt with Chain of Thought Placeholder>

**Output JSON keys & descriptors:**
1. **CoT**: Your step-by-step observations...
2. **Category**: ...
... (rest of the detailed user prompt with CoT)
"""
    else:
        system_prompt = "<system prompt without Chain of Thought Placeholder>"
        user_prompt = """
<user prompt without Chain of Thought Placeholder>

**Output JSON keys & descriptors:**
1. **Category**: ...
... (rest of the detailed user prompt without CoT)
"""
    return system_prompt, user_prompt