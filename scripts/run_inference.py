import argparse
from vehicle_inference.config import get_prompts
from vehicle_inference.vlms import QwenVLM, GemmaVLM
from vehicle_inference.pipeline import process_images
import gdown

def main():
    parser = argparse.ArgumentParser(description="Vehicle Inference CLI")
    parser.add_argument('--model', type=str, required=True, choices=['qwen', 'gemma'], help='VLM model to use')
    parser.add_argument('--input-folder', type=str, required=True, help='Input images folder')
    parser.add_argument('--output-folder', type=str, required=True, help='Output JSON folder')
    parser.add_argument('--use-finetuned', action='store_true', help='Use finetuned model')
    parser.add_argument('--cot', action='store_true', help='Enable Chain of Thought')
    parser.add_argument('--batch-size', type=int, default=2, help='VLM batch size')
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='YOLO confidence threshold')
    parser.add_argument('--download-samples', action='store_true', help='Download sample images')
    args = parser.parse_args()

    if args.download_samples:
        gdown.download(id="17PI4UeX2tDR2YDbWsRdVKFCECR9eYQ4_", output="file.zip", quiet=False)
        os.system("unzip file.zip -d input_images")

    system_prompt, user_prompt = get_prompts(args.cot)
    if args.model == 'qwen':
        vlm = QwenVLM(args.use_finetuned, system_prompt, user_prompt)
    else:
        vlm = GemmaVLM(args.use_finetuned, system_prompt, user_prompt)

    process_images(
        args.input_folder,
        args.output_folder,
        vlm,
        args.conf_threshold,
        args.batch_size
    )

if __name__ == "__main__":
    main()