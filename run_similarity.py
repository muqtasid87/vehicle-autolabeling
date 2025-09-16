# File: run_similarity.py
from similarity.comparator import JSONComparator

if __name__ == "__main__":
    folder1 = r"D:\python_projects\contract\Gemma\Gemma_Finetuned_Inference_23-26-07-09-2025\json_outputs"  # Replace with your actual path
    folder2 = r"D:\python_projects\contract\Gemma\Gemma_Finetuned_Inference_23-26-07-09-2025\json_outputs"  # Replace with your actual path
    
    comparator = JSONComparator(folder1, folder2)
    results = comparator.process_all_files()
    comparator.save_results()
