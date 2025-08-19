# File: similarity/comparator.py
import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import datetime
import matplotlib.pyplot as plt
from .configs import WEIGHTS
from .utils import calculate_category_similarity, calculate_subcategory_similarity, str_sim, int_sim, bool_sim

@dataclass
class ComparisonResult:
    filename: str
    similarity_score: float
    individual_scores: Dict[str, float]

class JSONComparator:
    def __init__(self, folder1_path: str, folder2_path: str):
        self.folder1_path = Path(folder1_path)
        self.folder2_path = Path(folder2_path)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def find_matching_files(self) -> Dict[str, Tuple[str, str]]:
        folder1_files = {f.stem: f for f in self.folder1_path.glob("*.json")}
        folder2_files = {f.stem: f for f in self.folder2_path.glob("*.json")}
        matching_files = {}
        for filename in folder1_files.keys():
            if filename in folder2_files:
                matching_files[filename] = (str(folder1_files[filename]), str(folder2_files[filename]))
        self.logger.info(f"Found {len(matching_files)} matching file pairs")
        return matching_files

    def load_json_file(self, filepath: str) -> Optional[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None

    def extract_vlm_inferences(self, json_data: Dict) -> List[Dict]:
        vlm_inferences = []
        if not json_data or 'detections' not in json_data:
            return vlm_inferences
        for detection in json_data['detections']:
            inference = detection.get('vlm_fine_grained_inference')
            if isinstance(inference, dict):
                vlm_inferences.append(inference)
            elif inference is not None:
                self.logger.warning(f"Skipping vlm_fine_grained_inference as it's not a dictionary: {type(inference)}")
        return vlm_inferences

    def compare_vlm_inference(self, vlm1: Dict[str, Any], vlm2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        individual_scores = {}
        # vehicle_info
        individual_scores['category'] = calculate_category_similarity(
            vlm1.get('vehicle_info', {}).get('category', 'Unclassified'),
            vlm2.get('vehicle_info', {}).get('category', 'Unclassified')
        )
        individual_scores['subcategory'] = calculate_subcategory_similarity(
            vlm1.get('vehicle_info', {}).get('subcategory', 'Unclassified'),
            vlm2.get('vehicle_info', {}).get('subcategory', 'Unclassified')
        )
        individual_scores['country'] = str_sim(
            vlm1.get('vehicle_info', {}).get('country', ''),
            vlm2.get('vehicle_info', {}).get('country', '')
        )
        individual_scores['brand'] = str_sim(
            vlm1.get('vehicle_info', {}).get('brand', ''),
            vlm2.get('vehicle_info', {}).get('brand', '')
        )
        individual_scores['model'] = str_sim(
            vlm1.get('vehicle_info', {}).get('model', ''),
            vlm2.get('vehicle_info', {}).get('model', '')
        )
        individual_scores['color'] = str_sim(
            vlm1.get('vehicle_info', {}).get('color', ''),
            vlm2.get('vehicle_info', {}).get('color', '')
        )
        individual_scores['operator'] = str_sim(
            vlm1.get('vehicle_info', {}).get('operator', ''),
            vlm2.get('vehicle_info', {}).get('operator', '')
        )
        individual_scores['number_of_seats'] = int_sim(
            vlm1.get('vehicle_info', {}).get('number_of_seats', 0),
            vlm2.get('vehicle_info', {}).get('number_of_seats', 0),
            max_diff=20
        )
        # mechanical
        individual_scores['number_of_wheels_visible'] = int_sim(
            vlm1.get('mechanical', {}).get('number_of_wheels_visible', 0),
            vlm2.get('mechanical', {}).get('number_of_wheels_visible', 0),
            max_diff=10
        )
        individual_scores['number_of_axles_inferred'] = int_sim(
            vlm1.get('mechanical', {}).get('number_of_axles_inferred', 0),
            vlm2.get('mechanical', {}).get('number_of_axles_inferred', 0),
            max_diff=6
        )
        individual_scores['number_of_axles_raised'] = int_sim(
            vlm1.get('mechanical', {}).get('number_of_axles_raised', 0),
            vlm2.get('mechanical', {}).get('number_of_axles_raised', 0),
            max_diff=3
        )
        individual_scores['truck_trailer_labels_visible'] = bool_sim(
            vlm1.get('mechanical', {}).get('truck_trailer_labels_visible', False),
            vlm2.get('mechanical', {}).get('truck_trailer_labels_visible', False)
        )
        individual_scores['cargo_present'] = bool_sim(
            vlm1.get('mechanical', {}).get('cargo_present', False),
            vlm2.get('mechanical', {}).get('cargo_present', False)
        )
        # attributes
        individual_scores['is_taxi'] = bool_sim(
            vlm1.get('attributes', {}).get('is_taxi', False),
            vlm2.get('attributes', {}).get('is_taxi', False)
        )
        individual_scores['is_bus'] = bool_sim(
            vlm1.get('attributes', {}).get('is_bus', False),
            vlm2.get('attributes', {}).get('is_bus', False)
        )
        individual_scores['bus_type'] = str_sim(
            vlm1.get('attributes', {}).get('bus_type', ''),
            vlm2.get('attributes', {}).get('bus_type', '')
        )
        individual_scores['is_emergency_vehicle'] = bool_sim(
            vlm1.get('attributes', {}).get('is_emergency_vehicle', False),
            vlm2.get('attributes', {}).get('is_emergency_vehicle', False)
        )
        individual_scores['is_electric'] = bool_sim(
            vlm1.get('attributes', {}).get('is_electric', False),
            vlm2.get('attributes', {}).get('is_electric', False)
        )
        # Calculate overall score
        total_weight = sum(WEIGHTS.values())
        overall_score = sum(WEIGHTS.get(k, 0) * v for k, v in individual_scores.items()) / total_weight
        return overall_score, individual_scores

    def compare_files(self, filename: str, file1_path: str, file2_path: str) -> ComparisonResult:
        json1 = self.load_json_file(file1_path)
        json2 = self.load_json_file(file2_path)
        if json1 is None or json2 is None:
            return ComparisonResult(filename=filename, similarity_score=0.0, individual_scores={})
        vlm1_list = self.extract_vlm_inferences(json1)
        vlm2_list = self.extract_vlm_inferences(json2)
        if not vlm1_list or not vlm2_list:
            return ComparisonResult(filename=filename, similarity_score=0.0, individual_scores={})
        all_scores = []
        all_individual_scores = {}
        max_comparisons = min(len(vlm1_list), len(vlm2_list))
        for i in range(max_comparisons):
            score, individual = self.compare_vlm_inference(vlm1_list[i], vlm2_list[i])
            all_scores.append(score)
            for key, val in individual.items():
                if key not in all_individual_scores:
                    all_individual_scores[key] = []
                all_individual_scores[key].append(val)
        avg_similarity = np.mean(all_scores) if all_scores else 0.0
        avg_individual_scores = {k: np.mean(v) for k, v in all_individual_scores.items()}
        return ComparisonResult(
            filename=filename,
            similarity_score=avg_similarity,
            individual_scores=avg_individual_scores
        )

    def process_all_files(self) -> List[ComparisonResult]:
        matching_files = self.find_matching_files()
        results = []
        for filename, (file1_path, file2_path) in matching_files.items():
            self.logger.info(f"Processing {filename}")
            result = self.compare_files(filename, file1_path, file2_path)
            results.append(result)
        folder1_files = set(f.stem for f in self.folder1_path.glob("*.json"))
        folder2_files = set(f.stem for f in self.folder2_path.glob("*.json"))
        only_in_folder1 = folder1_files - folder2_files
        only_in_folder2 = folder2_files - folder1_files
        for filename in only_in_folder1:
            results.append(ComparisonResult(filename=filename, similarity_score=0.0, individual_scores={}))
        for filename in only_in_folder2:
            results.append(ComparisonResult(filename=filename, similarity_score=0.0, individual_scores={}))
        self.results = results
        return results

    def save_results(self):
        if not self.results:
            self.logger.warning("No results to save.")
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("runs") / f"Similarity_Results_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_data = []
        for result in self.results:
            row = {'filename': result.filename, 'similarity_score': round(result.similarity_score, 4)}
            for key, score in result.individual_scores.items():
                row[f'{key}_similarity'] = round(score, 4)
            csv_data.append(row)
        df = pd.DataFrame(csv_data)
        csv_path = run_dir / "comparison_results.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")
        plt.figure(figsize=(10, 6))
        plt.hist(df['similarity_score'], bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.axvline(0.25, color='red', linestyle='dashed', linewidth=1.5, label='0.25')
        plt.axvline(0.50, color='green', linestyle='dashed', linewidth=1.5, label='0.50')
        plt.axvline(0.75, color='blue', linestyle='dashed', linewidth=1.5, label='0.75')
        plt.legend()
        plt.grid(axis='both', alpha=0.75)
        plot_path = run_dir / "similarity_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Histogram saved to {plot_path}")