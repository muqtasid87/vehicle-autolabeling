from inference.predictor import VlmPredictor

def run_inference(
    model_name: str,
    input_folder: str,
    lora_adapter_path: str = None,
    yolo_model_path: str = 'best_avc_v5.pt',
    batch_size: int = 4,
    use_lora: bool = True
):
    """
    Runs the inference pipeline using a VLM (with or without fine-tuned LoRA adapters).

    Args:
        model_name (str): The model to use for inference. Supported: 'qwen', 'gemma'.
        input_folder (str): The folder containing images to process.
        lora_adapter_path (str, optional): Path to the locally saved LoRA adapters. 
                                         Only needed if use_lora=True.
        yolo_model_path (str): Path to the YOLOv5 model file. Defaults to 'best_avc_v5.pt'.
        batch_size (int): The batch size for VLM inference. Defaults to 4.
        use_lora (bool): Whether to use LoRA adapters. If False, uses base model only. 
                        Defaults to True.
    """
    model_type = "fine-tuned" if use_lora else "base"
    print(f"🚀 Starting inference with {model_type} {model_name} model")
    
    predictor = VlmPredictor(
        model_name=model_name,
        input_folder=input_folder,
        lora_adapter_path=lora_adapter_path,
        yolo_model_path=yolo_model_path,
        batch_size=batch_size,
        use_lora=use_lora
    )
    output_path = predictor.run()
    print(f"✅ Inference complete!")
    print(f"📁 Results, logs, and JSON files are saved in: {output_path}")

if __name__ == '__main__':
    # --- EXAMPLE USAGE ---
    
    # Option 1: Run with LoRA adapters 
    # run_inference(
    #     model_name='qwen',  # or 'gemma'
    #     input_folder='sample_dataset/images',
    #     lora_adapter_path='/runs/Qwen_Finetuning_04-05-18-08-2025/lora_model',
    #     yolo_model_path='best_avc_v5.pt',
    #     batch_size=2,
    #     use_lora=True
    # )
    
    # Option 2: Run with base model only (no LoRA)
    run_inference(
        model_name='qwen',  # or 'gemma'
        input_folder='sample_dataset/images',
        lora_adapter_path=None,  
        yolo_model_path='best_avc_v5.pt',
        batch_size=2,
        use_lora=False  # This will use the base model only
    )