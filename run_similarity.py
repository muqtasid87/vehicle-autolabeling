# File: run_similarity.py
from similarity.comparator import JSONComparator

if __name__ == "__main__":
    folder1 = r""  # Replace with your actual path
    folder2 = r""  # Replace with your actual path
    
    comparator = JSONComparator(folder1, folder2)
    results = comparator.process_all_files()
    comparator.save_results()