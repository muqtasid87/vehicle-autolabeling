# src/vlm_inspector/cli.py

import argparse
from .pipeline import InferencePipeline

def main():
    parser = argparse.ArgumentParser(description="Run vehicle detection and VLM analysis.")
    parser.add_argument(
        "--input-dir", required=True, type=str, help="Path to the folder containing input images."
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Path to the folder where JSON outputs will be saved."
    )
    parser.add_argument(
        "--vlm-model", type=str, default="qwen", choices=["qwen", "gemma"],
        help="The VLM model to use for inference."
    )
    parser.add_argument(
        "--yolo-model-path", type=str, default="./models/best_avc_v5.pt",
        help="Path to the YOLOv5 model weights file."
    )
    parser.add_argument(
        "--vlm-batch-size", type=int, default=4, help="Batch size for VLM inference."
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.1, help="Confidence threshold for YOLOv5 detection."
    )
    parser.add_argument(
        "--use-finetuned", action=argparse.BooleanOptionalAction, default=True,
        help="Use the fine-tuned LoRA adapters. Use --no-use-finetuned to disable."
    )
    parser.add_argument(
        "--use-cot", action=argparse.BooleanOptionalAction, default=True,
        help="Use Chain of Thought in prompts. Use --no-use-cot to disable."
    )

    args = parser.parse_args()

    config = {
        "input_folder": args.input_dir,
        "output_folder": args.output_dir,
        "vlm_model": args.vlm_model,
        "yolo_model_path": args.yolo_model_path,
        "vlm_batch_size": args.vlm_batch_size,
        "conf_threshold": args.conf_threshold,
        "use_finetuned": args.use_finetuned,
        "use_cot": args.use_cot
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    pipeline = InferencePipeline(config)
    pipeline.run(config['input_folder'], config['output_folder'])

if __name__ == "__main__":
    main()