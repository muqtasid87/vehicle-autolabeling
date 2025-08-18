import unsloth
from finetuning.qwen_tuner import QwenTuner
from finetuning.gemma_tuner import GemmaTuner

def run_finetuning(
    model_name: str,
    input_folder: str,
    images_subfolder: str = "images",
    json_subfolder: str = "annotations",
    hyperparameters: dict = None
):
    """
    Starts the fine-tuning process for a specified model.

    Args:
        model_name (str): The model to fine-tune. Supported: 'qwen', 'gemma'.
        input_folder (str): The root directory containing the dataset.
        images_subfolder (str): Subdirectory for images within the input folder.
        json_subfolder (str): Subdirectory for JSON annotations within the input folder.
        hyperparameters (dict, optional): A dictionary of training arguments to override defaults.
    """
    print(f"🚀 Starting fine-tuning for model: {model_name}")

    if hyperparameters is None:
        hyperparameters = {} # Use defaults if none are provided

    if model_name.lower() == 'qwen':
        tuner = QwenTuner(input_folder, images_subfolder, json_subfolder, hyperparameters)
    elif model_name.lower() == 'gemma':
        tuner = GemmaTuner(input_folder, images_subfolder, json_subfolder, hyperparameters)
    else:
        raise ValueError("Unsupported model_name. Choose 'qwen' or 'gemma'.")

    tuner.train()
    print("✅ Fine-tuning complete!")

if __name__ == '__main__':
# ---------------------------------
# --- Fine-Tuning Example ---
# ---------------------------------

    # Define custom hyperparameters for training. Any SFTConfig argument can be overridden.
    custom_hyperparams = {
        "num_train_epochs": 1,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 2,
    }

    run_finetuning(
        model_name='gemma',  # or 'qwen'
        input_folder='sample_dataset',
        images_subfolder='images', # Assumes dataset folder has 'images' and 'annotations' subfolders
        json_subfolder='annotations',
        hyperparameters=custom_hyperparams
    )
