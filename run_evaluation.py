from similarity.evaluator import LLMEvaluator
# --- Configuration ---
# Folder containing ground truth JSON files
GT_FOLDER = r"D:\python_projects\contract\data_cleaning\dataset\testing\annotations"
# Folder containing the LLM inference JSON files
LLM_FOLDER = r"D:\python_projects\contract\Qwen\Qwen_Finetuned_Inference_11-31-16-09-2025\json_outputs"
# IoU threshold for a detection to be considered a potential match for mAP
IOU_THRESHOLD = 0.5

if __name__ == '__main__':
    # Initialize the evaluator
    
    ev = LLMEvaluator(GT_FOLDER, LLM_FOLDER, iou_threshold_map=0.5, similarity_iou_threshold=0.75)
    results, map_summary = ev.process_all_files()
    ev.save_results()

