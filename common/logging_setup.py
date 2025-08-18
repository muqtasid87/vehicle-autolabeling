import logging
import os
from datetime import datetime

def setup_logging_and_dir(base_folder_name: str, model_name: str):
    """
    Creates a timestamped directory inside a main 'runs' folder and sets up logging.

    Args:
        base_folder_name (str): The base name for the output folder (e.g., "Finetuning", "Inference").
        model_name (str): The name of the model being used (e.g., "Qwen", "Gemma").

    Returns:
        str: The path to the created output directory.
    """
    # --- CHANGE STARTS HERE ---

    # 1. Define the main 'runs' directory
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)

    # 2. Create the timestamped folder *inside* the 'runs' directory
    timestamp = datetime.now().strftime("%H-%M-%d-%m-%Y")
    output_dir_name = f"{model_name}_{base_folder_name}_{timestamp}"
    output_dir_path = os.path.join(runs_dir, output_dir_name)
    os.makedirs(output_dir_path, exist_ok=True)

    # --- CHANGE ENDS HERE ---

    log_file = os.path.join(output_dir_path, f'{base_folder_name.lower()}.log')

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Disable propagation for 'ultralytics' to avoid duplicate logging
    logging.getLogger('ultralytics').propagate = False
    
    logging.info(f"Output directory created at: {output_dir_path}")
    return output_dir_path