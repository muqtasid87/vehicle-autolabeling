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
    def __init__(self, folder1_path: str, folder2_path: str, verbose: bool = True):
        self.folder1_path = Path(folder1_path)
        self.folder2_path = Path(folder2_path)
        self.logger = logging.getLogger(__name__)
        # configure logger
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.results: List[ComparisonResult] = []

    def find_matching_files(self) -> Dict[str, Tuple[str, str]]:
        folder1_files = {f.stem: f for f in self.folder1_path.glob("*.json")}
        folder2_files = {f.stem: f for f in self.folder2_path.glob("*.json")}
        self.logger.debug(f"Folder1 files (stem -> path): { {k:str(v) for k,v in folder1_files.items()} }")
        self.logger.debug(f"Folder2 files (stem -> path): { {k:str(v) for k,v in folder2_files.items()} }")
        matching_files = {}
        for filename in folder1_files.keys():
            if filename in folder2_files:
                matching_files[filename] = (str(folder1_files[filename]), str(folder2_files[filename]))
        self.logger.info(f"Found {len(matching_files)} matching file pairs: {list(matching_files.keys())}")
        return matching_files

    def load_json_file(self, filepath: str) -> Optional[Dict]:
        self.logger.debug(f"Loading JSON file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # log top-level keys and type info
            if isinstance(data, dict):
                self.logger.debug(f"Top-level keys in {filepath}: {list(data.keys())}")
            else:
                self.logger.debug(f"Loaded JSON is of type {type(data)}")
            return data
        except Exception as e:
            self.logger.exception(f"Error loading {filepath}: {e}")
            return None

    def extract_vlm_inferences(self, json_data: Dict) -> List[Dict]:
        vlm_inferences = []
        if not json_data:
            self.logger.debug("extract_vlm_inferences: json_data is empty or None")
            return vlm_inferences

        # Common entry points
        if 'detections' not in json_data:
            # sometimes the detections might be nested or named differently
            self.logger.warning("'detections' key not found in JSON root. Trying to find list of detections by searching for keys containing 'detection' or lists of dicts...")
            # try to find any list value that looks like detections
            for k, v in json_data.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    self.logger.debug(f"Assuming key '{k}' contains detection-like list (type: list of dicts).")
                    json_data = { 'detections': v }
                    break

        detections = json_data.get('detections') if isinstance(json_data, dict) else None
        if not isinstance(detections, list):
            self.logger.warning("No 'detections' list could be found after fallback. Returning empty inference list.")
            return vlm_inferences

        self.logger.debug(f"Found {len(detections)} detections")
        for idx, detection in enumerate(detections):
            if not isinstance(detection, dict):
                self.logger.debug(f"Skipping detection at index {idx} because it's not a dict (type: {type(detection)})")
                continue
            # Candidate keys that may hold the fine-grained vlm inference
            candidates = [
                'vlm_fine_grained_inference', 'vlm_inference', 'vlm_fine_grained_inferences',
                'inference', 'fine_grained_inference'
            ]
            inference = None
            for key in candidates:
                if key in detection:
                    candidate = detection.get(key)
                    if isinstance(candidate, dict):
                        inference = candidate
                        self.logger.debug(f"Using inference from key '{key}' in detection index {idx}")
                        break
                    elif isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
                        self.logger.debug(f"Key '{key}' contained a list of inferences; taking the first dict for detection index {idx}")
                        inference = candidate[0]
                        break
                    else:
                        self.logger.debug(f"Key '{key}' exists but is not a dict/list-of-dict (type: {type(candidate)})")

            # If none of the known keys matched, try to find any dict value that looks like a vlm inference
            if inference is None:
                for k, v in detection.items():
                    if isinstance(v, dict) and 'vehicle_info' in v:
                        inference = v
                        self.logger.debug(f"Found inference by searching for 'vehicle_info' in key '{k}' at detection index {idx}")
                        break

            if inference is None:
                # As a last resort, check if detection itself *looks* like an inference (has vehicle_info)
                if 'vehicle_info' in detection:
                    inference = detection
                    self.logger.debug(f"Detection at index {idx} appears to be the inference itself (contains 'vehicle_info')")

            if inference is None:
                # Log available keys for debugging
                self.logger.debug(f"Skipping detection index {idx} - no inference found. Available keys: {list(detection.keys())}")
                continue

            # final sanity check
            if not isinstance(inference, dict):
                self.logger.warning(f"Inference found at detection index {idx} is not a dict (type: {type(inference)}). Skipping.")
                continue

            vlm_inferences.append(inference)
        self.logger.info(f"Extracted {len(vlm_inferences)} vlm_inference dict(s)")
        return vlm_inferences

    def compare_vlm_inference(self, vlm1: Dict[str, Any], vlm2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        individual_scores = {}
        try:
            # vehicle_info
            cat1 = vlm1.get('vehicle_info', {}).get('category', 'Unclassified')
            cat2 = vlm2.get('vehicle_info', {}).get('category', 'Unclassified')
            individual_scores['category'] = calculate_category_similarity(cat1, cat2)
            self.logger.debug(f"category: '{cat1}' vs '{cat2}' -> {individual_scores['category']}")

            sub1 = vlm1.get('vehicle_info', {}).get('subcategory', 'Unclassified')
            sub2 = vlm2.get('vehicle_info', {}).get('subcategory', 'Unclassified')
            individual_scores['subcategory'] = calculate_subcategory_similarity(sub1, sub2)
            self.logger.debug(f"subcategory: '{sub1}' vs '{sub2}' -> {individual_scores['subcategory']}")

            individual_scores['country'] = str_sim(
                vlm1.get('vehicle_info', {}).get('country', ''),
                vlm2.get('vehicle_info', {}).get('country', '')
            )
            self.logger.debug(f"country -> {individual_scores['country']}")

            individual_scores['brand'] = str_sim(
                vlm1.get('vehicle_info', {}).get('brand', ''),
                vlm2.get('vehicle_info', {}).get('brand', '')
            )
            self.logger.debug(f"brand -> {individual_scores['brand']}")

            individual_scores['model'] = str_sim(
                vlm1.get('vehicle_info', {}).get('model', ''),
                vlm2.get('vehicle_info', {}).get('model', '')
            )
            self.logger.debug(f"model -> {individual_scores['model']}")

            individual_scores['color'] = str_sim(
                vlm1.get('vehicle_info', {}).get('color', ''),
                vlm2.get('vehicle_info', {}).get('color', '')
            )
            self.logger.debug(f"color -> {individual_scores['color']}")

            individual_scores['operator'] = str_sim(
                vlm1.get('vehicle_info', {}).get('operator', ''),
                vlm2.get('vehicle_info', {}).get('operator', '')
            )
            self.logger.debug(f"operator -> {individual_scores['operator']}")

            individual_scores['number_of_seats'] = int_sim(
                vlm1.get('vehicle_info', {}).get('number_of_seats', 0),
                vlm2.get('vehicle_info', {}).get('number_of_seats', 0),
                max_diff=20
            )
            self.logger.debug(f"number_of_seats -> {individual_scores['number_of_seats']}")

            # mechanical
            individual_scores['number_of_wheels_visible'] = int_sim(
                vlm1.get('mechanical', {}).get('number_of_wheels_visible', 0),
                vlm2.get('mechanical', {}).get('number_of_wheels_visible', 0),
                max_diff=10
            )
            self.logger.debug(f"number_of_wheels_visible -> {individual_scores['number_of_wheels_visible']}")

            individual_scores['number_of_axles_inferred'] = int_sim(
                vlm1.get('mechanical', {}).get('number_of_axles_inferred', 0),
                vlm2.get('mechanical', {}).get('number_of_axles_inferred', 0),
                max_diff=6
            )
            self.logger.debug(f"number_of_axles_inferred -> {individual_scores['number_of_axles_inferred']}")

            individual_scores['number_of_axles_raised'] = int_sim(
                vlm1.get('mechanical', {}).get('number_of_axles_raised', 0),
                vlm2.get('mechanical', {}).get('number_of_axles_raised', 0),
                max_diff=3
            )
            self.logger.debug(f"number_of_axles_raised -> {individual_scores['number_of_axles_raised']}")

            individual_scores['truck_trailer_labels_visible'] = bool_sim(
                vlm1.get('mechanical', {}).get('truck_trailer_labels_visible', False),
                vlm2.get('mechanical', {}).get('truck_trailer_labels_visible', False)
            )
            self.logger.debug(f"truck_trailer_labels_visible -> {individual_scores['truck_trailer_labels_visible']}")

            individual_scores['cargo_present'] = bool_sim(
                vlm1.get('mechanical', {}).get('cargo_present', False),
                vlm2.get('mechanical', {}).get('cargo_present', False)
            )
            self.logger.debug(f"cargo_present -> {individual_scores['cargo_present']}")

            # attributes
            individual_scores['is_taxi'] = bool_sim(
                vlm1.get('attributes', {}).get('is_taxi', False),
                vlm2.get('attributes', {}).get('is_taxi', False)
            )
            self.logger.debug(f"is_taxi -> {individual_scores['is_taxi']}")

            individual_scores['is_bus'] = bool_sim(
                vlm1.get('attributes', {}).get('is_bus', False),
                vlm2.get('attributes', {}).get('is_bus', False)
            )
            self.logger.debug(f"is_bus -> {individual_scores['is_bus']}")

            individual_scores['bus_type'] = str_sim(
                vlm1.get('attributes', {}).get('bus_type', ''),
                vlm2.get('attributes', {}).get('bus_type', '')
            )
            self.logger.debug(f"bus_type -> {individual_scores['bus_type']}")

            individual_scores['is_emergency_vehicle'] = bool_sim(
                vlm1.get('attributes', {}).get('is_emergency_vehicle', False),
                vlm2.get('attributes', {}).get('is_emergency_vehicle', False)
            )
            self.logger.debug(f"is_emergency_vehicle -> {individual_scores['is_emergency_vehicle']}")

            individual_scores['is_electric'] = bool_sim(
                vlm1.get('attributes', {}).get('is_electric', False),
                vlm2.get('attributes', {}).get('is_electric', False)
            )
            self.logger.debug(f"is_electric -> {individual_scores['is_electric']}")

            # Calculate overall score
            total_weight = sum(WEIGHTS.values())
            self.logger.debug(f"Total weight sum: {total_weight}")
            # sum only weights for keys present in individual_scores
            weighted_sum = 0.0
            for k, v in individual_scores.items():
                w = WEIGHTS.get(k, 0)
                weighted_sum += w * v
                self.logger.debug(f"key: {k}, weight: {w}, value: {v}, contrib: {w*v}")

            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            self.logger.debug(f"Overall score (weighted): {overall_score}")
            return overall_score, individual_scores
        except Exception as e:
            self.logger.exception(f"Exception while comparing vlm inferences: {e}")
            return 0.0, {}

    def compare_files(self, filename: str, file1_path: str, file2_path: str) -> ComparisonResult:
        self.logger.info(f"Comparing file pair: {filename}")
        json1 = self.load_json_file(file1_path)
        json2 = self.load_json_file(file2_path)
        if json1 is None or json2 is None:
            self.logger.error("One of the JSON files failed to load. Returning zero similarity for this file.")
            return ComparisonResult(filename=filename, similarity_score=0.0, individual_scores={})
        vlm1_list = self.extract_vlm_inferences(json1)
        vlm2_list = self.extract_vlm_inferences(json2)
        self.logger.debug(f"vlm1_list length: {len(vlm1_list)}, vlm2_list length: {len(vlm2_list)})")
        if not vlm1_list or not vlm2_list:
            self.logger.error("No vlm inferences found in one or both files. Returning zero similarity for this file.")
            return ComparisonResult(filename=filename, similarity_score=0.0, individual_scores={})
        all_scores = []
        all_individual_scores = {}
        max_comparisons = min(len(vlm1_list), len(vlm2_list))
        self.logger.debug(f"Will compare up to {max_comparisons} pairs (min of lengths)")
        for i in range(max_comparisons):
            self.logger.debug(f"Comparing inference index {i}")
            score, individual = self.compare_vlm_inference(vlm1_list[i], vlm2_list[i])
            self.logger.debug(f"Score for index {i}: {score}")
            all_scores.append(score)
            for key, val in individual.items():
                if key not in all_individual_scores:
                    all_individual_scores[key] = []
                all_individual_scores[key].append(val)
        avg_similarity = float(np.mean(all_scores)) if all_scores else 0.0
        avg_individual_scores = {k: float(np.mean(v)) for k, v in all_individual_scores.items()}
        self.logger.info(f"Avg similarity for {filename}: {avg_similarity}")
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
        # Plot histogram only if similarity_score column exists
        if 'similarity_score' in df.columns and not df['similarity_score'].isnull().all():
            plt.figure(figsize=(10, 6))
            plt.hist(df['similarity_score'], bins=20, edgecolor='black', alpha=0.7)
            plt.title('Distribution of Similarity Scores')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.axvline(0.25, linestyle='dashed', linewidth=1.5, label='0.25')
            plt.axvline(0.50, linestyle='dashed', linewidth=1.5, label='0.50')
            plt.axvline(0.75, linestyle='dashed', linewidth=1.5, label='0.75')
            plt.legend()
            plt.grid(axis='both', alpha=0.75)
            plot_path = run_dir / "similarity_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Histogram saved to {plot_path}")
        else:
            self.logger.warning("No valid similarity_score column to plot histogram.")