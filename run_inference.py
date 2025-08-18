import unsloth
from inference.predictor import VlmPredictor

def run_inference(
    model_name: str,
    input_folder: str,
    lora_adapter_path: str,
    yolo_model_path: str = 'best_avc_v5.pt',
    batch_size: int = 4
):
    """
    Runs the inference pipeline using a fine-tuned VLM.

    Args:
        model_name (str): The model to use for inference. Supported: 'qwen', 'gemma'.
        input_folder (str): The folder containing images to process.
        lora_adapter_path (str): Path to the locally saved LoRA adapters.
        yolo_model_path (str): Path to the YOLOv5 model file. Defaults to 'best_avc_v5.pt'.
        batch_size (int): The batch size for VLM inference. Defaults to 4.
    """
    print(f"🚀 Starting inference with model: {model_name}")
    predictor = VlmPredictor(
        model_name=model_name,
        input_folder=input_folder,
        lora_adapter_path=lora_adapter_path,
        yolo_model_path=yolo_model_path,
        batch_size=batch_size
    )
    output_path = predictor.run()
    print(f"✅ Inference complete!")
    print(f"📁 Results, logs, and JSON files are saved in: {output_path}")


if __name__ == '__main__':
# ---------------------------------
# --- Inference Example ---
# ---------------------------------
    
    # run_inference(
    #     model_name='qwen',  # or 'gemma'
    #     input_folder='vlm_pipeline\sample_dataset\images',
    #     lora_adapter_path='/path/to/your/local_lora_model', # This is the folder saved after finetuning
    #     yolo_model_path='best_avc_v5.pt', # Make sure this file is accessible
    #     batch_size=8
    # )