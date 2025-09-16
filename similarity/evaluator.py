# File: similarity/evaluator.py
"""
Robust LLM evaluator for VLM-style outputs vs ground truth annotations.

- Matches files where GT names end with "_annotations.json" while LLM inferences use the base stem.
- Uses IoU matching (default 0.5 for mAP; uses 0.75 for similarity comparisons as requested).
- Computes per-field similarity using functions from similarity.utils (calculate_category_similarity, etc).
- Computes per-class AP and mean AP (mAP), saves detailed CSVs and plots.
- Lots of DEBUG/INFO logging, plus a rotating log file stored in run directory.
"""
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import datetime
import matplotlib.pyplot as plt
import re

from .configs import WEIGHTS
from .utils import calculate_category_similarity, calculate_subcategory_similarity, str_sim, int_sim, bool_sim

# ----------------------------
# Helpers
# ----------------------------
def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate IoU between two boxes in [x1,y1,x2,y2] format.
    Boxes may contain floats or ints. Ensures numeric casting and valid ordering.
    """
    try:
        ax1, ay1, ax2, ay2 = [float(x) for x in boxA]
        bx1, by1, bx2, by2 = [float(x) for x in boxB]
    except Exception:
        return 0.0

    # normalize coordinates if given reversed
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0

    boxA_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    boxB_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = boxA_area + boxB_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def _normalize_placeholders(obj):
    """Replace common placeholder strings with empty string and lower/strip textual fields."""
    if isinstance(obj, dict):
        return {k: _normalize_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, str):
        s = obj.strip()
        if s.lower() in ("", "empty string", "none", "nan", "null"):
            return ""
        return s
    return obj

# ----------------------------
# Data classes
# ----------------------------
@dataclass
class EvaluationResult:
    filename: str
    similarity_score: float
    individual_scores: Dict[str, float]
    num_gt_objects: int
    num_llm_detections: int
    num_matched_pairs: int

@dataclass
class MAPResult:
    category: str
    precision: float
    recall: float
    ap: float
    tp: int
    fp: int
    fn: int

# ----------------------------
# Evaluator
# ----------------------------
class LLMEvaluator:
    def __init__(self, gt_folder_path: str, llm_folder_path: str, iou_threshold_map: float = 0.5, similarity_iou_threshold: float = 0.75):
        self.gt_folder = Path(gt_folder_path)
        self.llm_folder = Path(llm_folder_path)
        self.iou_threshold_map = float(iou_threshold_map)
        self.similarity_iou_threshold = float(similarity_iou_threshold)
        self.results: List[EvaluationResult] = []
        self.map_data: List[Dict[str, Any]] = []
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path("runs") / f"LLM_Evaluation_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # logger: file + console
        log_file_path = self.run_dir / "evaluation.log"
        self.logger = logging.getLogger(f"LLMEvaluator_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fh = RotatingFileHandler(str(log_file_path), maxBytes=5*1024*1024, backupCount=3)
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.logger.info(f"Run directory: {self.run_dir}")
        self.logger.info(f"Log file: {log_file_path}")

    # ----------------------------
    # Matching files
    # ----------------------------
    def find_matching_files(self) -> Dict[str, Tuple[Path, Path]]:
        """
        Matches GT (ending in _annotations.json) to LLM inferences. Returns dict: stem -> (gt_path, llm_path)
        """
        self.logger.info("Starting file matching...")
        gt_files = {}
        for f in sorted(self.gt_folder.glob("*_annotations.json")):
            stem = f.stem  # e.g. "image1_annotations"
            if stem.endswith("_annotations"):
                stem = stem[:-len("_annotations")]
            gt_files[stem] = f

        llm_files = {f.stem: f for f in sorted(self.llm_folder.glob("*.json"))}

        gt_stems = set(gt_files.keys())
        llm_stems = set(llm_files.keys())
        matches = gt_stems & llm_stems
        matched = {s: (gt_files[s], llm_files[s]) for s in matches}
        self.logger.info(f"Found {len(matched)} matched files.")
        if matched:
            for s in sorted(matched.keys()):
                self.logger.debug(f" MATCH: {s} -> GT={matched[s][0].name}, LLM={matched[s][1].name}")

        # log unmatched
        unmatched_gt = gt_stems - llm_stems
        if unmatched_gt:
            self.logger.warning(f"{len(unmatched_gt)} GT files without LLM counterpart:")
            for s in sorted(unmatched_gt):
                self.logger.warning(f"  - {gt_files[s].name}")

        unmatched_llm = llm_stems - gt_stems
        if unmatched_llm:
            self.logger.warning(f"{len(unmatched_llm)} LLM files without GT counterpart:")
            for s in sorted(unmatched_llm):
                self.logger.warning(f"  - {llm_files[s].name}")

        return matched

    # ----------------------------
    # IO helpers
    # ----------------------------
    def load_json_file(self, filepath: Path) -> Optional[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.logger.debug(f"Loaded {filepath.name} keys: {list(data.keys())}")
            else:
                self.logger.debug(f"Loaded {filepath.name} type: {type(data)}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load JSON {filepath}: {e}")
            return None

    # ----------------------------
    # Extractors
    # ----------------------------
    def extract_gt_objects(self, json_data: Dict) -> List[Dict]:
        """
        Return list of GT vehicle objects with normalized bbox and flags for matching.
        Each item: {'id': idx, 'bbox': [x1,y1,x2,y2], 'data':obj, 'matched_for_map', 'matched_for_sim'}
        """
        if not json_data:
            return []
        objs = json_data.get('objects', [])
        self.logger.debug(f"GT file contains {len(objs)} objects total.")
        gt_list = []
        for i, obj in enumerate(objs):
            if obj.get('type') != 'vehicle':
                self.logger.debug(f"Skipping GT object index {i} because type != 'vehicle' ({obj.get('type')})")
                continue
            bbox = obj.get('bbox', None)
            if not bbox or len(bbox) < 4:
                self.logger.debug(f"Skipping GT object index {i} because bbox is invalid: {bbox}")
                continue
            # Cast to floats/ints
            bbox_norm = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            gt_list.append({
                'id': i,
                'bbox': bbox_norm,
                'data': _normalize_placeholders(obj),
                'matched_for_map': False,
                'matched_for_sim': False
            })
        self.logger.info(f"Filtered GT to {len(gt_list)} 'vehicle' objects for evaluation.")
        return gt_list

    def extract_llm_detections(self, json_data: Dict) -> List[Dict]:
        """
        Return list of LLM detections: {'id': idx, 'bbox': [x1,y1,x2,y2], 'data':vlm_inference_dict, 'confidence': float}
        Accepts 'bounding_box' as dict {x1,y1,x2,y2} and normalizes.
        """
        if not json_data:
            return []
        dets = json_data.get('detections', [])
        out = []
        for i, det in enumerate(dets):
            bbox_raw = det.get('bounding_box', {})
            # If bounding box present as dict with x1,y1,x2,y2
            bx = [
                bbox_raw.get('x1', None),
                bbox_raw.get('y1', None),
                bbox_raw.get('x2', None),
                bbox_raw.get('y2', None)
            ]
            if any(v is None for v in bx):
                self.logger.debug(f"LLM det #{i} bounding_box incomplete: {bbox_raw}. Skipping detection.")
                continue
            try:
                bx_norm = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
            except Exception:
                self.logger.debug(f"LLM det #{i} bounding_box contains non-numeric values: {bx}. Skipping.")
                continue

            confidence = float(det.get('confidence', 1.0))
            # 'vlm_inference' is expected; fallbacks allowed
            vlm = det.get('vlm_inference') or det.get('vlm_fine_grained_inference') or det.get('inference') or {}
            vlm = _normalize_placeholders(vlm)
            out.append({'id': i, 'bbox': bx_norm, 'data': vlm, 'confidence': confidence})
        self.logger.info(f"Extracted {len(out)} LLM detections.")
        return out

    # ----------------------------
    # Similarity comparison
    # ----------------------------
    def compare_vlm_inference(self, gt_data: Dict[str, Any], llm_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Compute per-field similarities and overall weighted similarity using WEIGHTS.
        Expects both gt_data and llm_data to contain a 'vehicle_info' dict at root (per your examples).
        """
        try:
            gt_v = gt_data.get('vehicle_info', {}) if isinstance(gt_data, dict) else {}
            ll_v = llm_data.get('vehicle_info', {}) if isinstance(llm_data, dict) else {}

            # normalize textual placeholders
            gt_v = _normalize_placeholders(gt_v)
            ll_v = _normalize_placeholders(ll_v)

            individual_scores = {}
            individual_scores['category'] = calculate_category_similarity(
                gt_v.get('category', 'Unclassified'),
                ll_v.get('category', 'Unclassified')
            )
            individual_scores['subcategory'] = calculate_subcategory_similarity(
                gt_v.get('subcategory', 'Unclassified'),
                ll_v.get('subcategory', 'Unclassified')
            )
            individual_scores['country'] = str_sim(gt_v.get('country', ''), ll_v.get('country', ''))
            individual_scores['brand'] = str_sim(gt_v.get('brand', ''), ll_v.get('brand', ''))
            individual_scores['model'] = str_sim(gt_v.get('model', ''), ll_v.get('model', ''))
            individual_scores['color'] = str_sim(gt_v.get('color', ''), ll_v.get('color', ''))
            individual_scores['operator'] = str_sim(gt_v.get('operator', ''), ll_v.get('operator', ''))
            individual_scores['number_of_seats'] = int_sim(
                gt_v.get('number_of_seats', 0),
                ll_v.get('number_of_seats', 0),
                max_diff=20
            )

            # mechanical & attributes only if present
            gt_m = gt_data.get('mechanical', {}) if isinstance(gt_data, dict) else {}
            ll_m = llm_data.get('mechanical', {}) if isinstance(llm_data, dict) else {}
            individual_scores['number_of_wheels_visible'] = int_sim(
                gt_m.get('number_of_wheels_visible', 0), ll_m.get('number_of_wheels_visible', 0), max_diff=10
            )
            individual_scores['number_of_axles_inferred'] = int_sim(
                gt_m.get('number_of_axles_inferred', 0), ll_m.get('number_of_axles_inferred', 0), max_diff=6
            )
            individual_scores['number_of_axles_raised'] = int_sim(
                gt_m.get('number_of_axles_raised', 0), ll_m.get('number_of_axles_raised', 0), max_diff=3
            )
            individual_scores['truck_trailer_labels_visible'] = bool_sim(
                gt_m.get('truck_trailer_labels_visible', False), ll_m.get('truck_trailer_labels_visible', False)
            )
            individual_scores['cargo_present'] = bool_sim(
                gt_m.get('cargo_present', False), ll_m.get('cargo_present', False)
            )

            gt_a = gt_data.get('attributes', {}) if isinstance(gt_data, dict) else {}
            ll_a = llm_data.get('attributes', {}) if isinstance(llm_data, dict) else {}
            individual_scores['is_taxi'] = bool_sim(gt_a.get('is_taxi', False), ll_a.get('is_taxi', False))
            individual_scores['is_bus'] = bool_sim(gt_a.get('is_bus', False), ll_a.get('is_bus', False))
            individual_scores['bus_type'] = str_sim(gt_a.get('bus_type', ''), ll_a.get('bus_type', ''))
            individual_scores['is_emergency_vehicle'] = bool_sim(
                gt_a.get('is_emergency_vehicle', False), ll_a.get('is_emergency_vehicle', False)
            )
            individual_scores['is_electric'] = bool_sim(gt_a.get('is_electric', False), ll_a.get('is_electric', False))

            total_weight = sum(WEIGHTS.values())
            if total_weight == 0:
                return 0.0, individual_scores
            weighted_sum = 0.0
            for k, v in individual_scores.items():
                w = WEIGHTS.get(k, 0.0)
                weighted_sum += w * v
            overall = weighted_sum / total_weight
            return float(overall), {k: float(v) for k, v in individual_scores.items()}
        except Exception as e:
            self.logger.exception(f"Exception in compare_vlm_inference: {e}")
            return 0.0, {}

    # ----------------------------
    # Evaluate pair of files
    # ----------------------------
    def evaluate_file_pair(self, filename: str, gt_path: Path, llm_path: Path) -> EvaluationResult:
        self.logger.info(f"Evaluating {filename} ...")
        gt_json = self.load_json_file(gt_path)
        llm_json = self.load_json_file(llm_path)
        gt_objs = self.extract_gt_objects(gt_json)
        llm_dets = self.extract_llm_detections(llm_json)

        if not gt_objs and not llm_dets:
            self.logger.warning("No GT vehicles and no LLM detections; skipping.")
            return EvaluationResult(filename, 0.0, {}, 0, 0, 0)

        # --- Similarity matching (IoU > similarity_iou_threshold) ---
        sim_scores = []
        sim_individual = {}
        self.logger.debug(f"Starting similarity matching using IoU > {self.similarity_iou_threshold:.3f}")
        for ldet in llm_dets:
            best_iou = -1.0
            best_gt = None
            for gt in gt_objs:
                iou = calculate_iou(ldet['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            if best_iou >= self.similarity_iou_threshold and best_gt is not None and not best_gt['matched_for_sim']:
                best_gt['matched_for_sim'] = True
                self.logger.debug(f"SIM match: LLM#{ldet['id']} <-> GT#{best_gt['id']} IoU={best_iou:.4f}")
                score, individual = self.compare_vlm_inference(best_gt['data'], ldet['data'])
                sim_scores.append(score)
                for k, v in individual.items():
                    sim_individual.setdefault(k, []).append(v)
            else:
                self.logger.debug(f"No SIM match for LLM#{ldet['id']} (best_iou={best_iou:.4f})")

        avg_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
        avg_individual_sim = {k: float(np.mean(v)) for k, v in sim_individual.items()} if sim_individual else {}

        # --- mAP bookkeeping (IoU > iou_threshold_map) ---
        # For each llm detection, find best GT (by IoU). If IoU>threshold and GT not used yet, mark TP if categories match else FP.
        self.logger.debug(f"Collecting mAP entries using IoU > {self.iou_threshold_map:.3f}")
        for ldet in llm_dets:
            best_iou = -1.0
            best_gt = None
            for gt in gt_objs:
                iou = calculate_iou(ldet['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            predicted_category = (ldet['data'].get('vehicle_info', {}).get('category') if isinstance(ldet['data'], dict) else None) or ""
            is_correct = False
            gt_cat = None
            if best_gt is not None and best_iou >= self.iou_threshold_map:
                gt_cat = best_gt['data'].get('vehicle_info', {}).get('category', '') if isinstance(best_gt['data'], dict) else ""
                # If GT already claimed for map, this prediction cannot count as TP
                if not best_gt['matched_for_map']:
                    if predicted_category == gt_cat:
                        is_correct = True
                        best_gt['matched_for_map'] = True
                    else:
                        is_correct = False
                else:
                    is_correct = False
            else:
                # No gt found with sufficient IoU => false positive
                is_correct = False

            self.map_data.append({
                'predicted_category': predicted_category,
                'gt_category': gt_cat,
                'is_correct': bool(is_correct),
                'confidence': float(ldet.get('confidence', 1.0)),
                'filename': filename
            })

        # Add FN entries count as unmatched GTs (these are used to compute num_gt per class)
        for gt in gt_objs:
            if not gt['matched_for_map']:
                # For clarity we do not append an extra "prediction" row here; we record GT counts separately in calculate_map
                self.logger.debug(f"GT not matched for mAP: GT#{gt['id']} category={gt['data'].get('vehicle_info', {}).get('category', '')}")

        return EvaluationResult(filename, avg_sim, avg_individual_sim, len(gt_objs), len(llm_dets), len(sim_scores))

    # ----------------------------
    # Calculate per-class AP and overall mAP
    # ----------------------------
    def calculate_map(self) -> List[MAPResult]:
        """
        Process self.map_data (list of dicts containing predicted_category, gt_category, is_correct, confidence, filename)
        and compute per-class AP, TP, FP, FN and return list of MAPResult.
        """
        if not self.map_data:
            self.logger.warning("No map_data to compute mAP.")
            return []

        df = pd.DataFrame(self.map_data)

        gt_category_counts: Dict[str, int] = {}

        processed_files = set(df['filename'].unique())
        for fname in processed_files:
            pass

        df_gt = df[df['gt_category'].astype(bool)]
        num_gt_per_cat = df_gt.groupby('gt_category').size().to_dict()

        results = []
        categories = set(df['predicted_category'].unique()) | set(df['gt_category'].unique())
        categories = {c for c in categories if c and c != 'FP_category'}

        for cat in sorted(categories):
            pred_df = df[df['predicted_category'] == cat].sort_values('confidence', ascending=False)
            if pred_df.empty:
                tp = 0
                fp = 0
                fn = int(num_gt_per_cat.get(cat, 0))
                ap = 0.0
                results.append(MAPResult(cat, 0.0, 0.0, ap, tp, fp, fn))
                self.logger.info(f"Category {cat}: no predictions, FN={fn}")
                continue

            tp_array = pred_df['is_correct'].astype(int).values
            fp_array = (1 - pred_df['is_correct'].astype(int)).values
            tps = np.cumsum(tp_array)
            fps = np.cumsum(fp_array)
            num_gt = int(num_gt_per_cat.get(cat, 0))
            if num_gt == 0:
                # no GT of this class in dataset -> skip AP computation (or define AP=NaN)
                ap = float('nan')
                prec = float(tps[-1]) / (tps[-1] + fps[-1]) if (tps[-1] + fps[-1]) > 0 else 0.0
                rec = float(tps[-1]) / max(1, num_gt)
                results.append(MAPResult(cat, prec, rec, ap, int(tps[-1]), int(fps[-1]), 0))
                self.logger.info(f"Category {cat}: no GT present; predictions => TP={int(tps[-1])}, FP={int(fps[-1])}")
                continue

            recalls = tps / float(num_gt)
            precisions = tps / (tps + fps + 1e-12)
            # add sentinel
            mpre = np.concatenate(([0.0], precisions, [0.0]))
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            # make precision monotonic
            for i in range(mpre.size - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            # compute AP as area under PR curve
            idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
            ap = float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
            prec = float(precisions[-1]) if precisions.size else 0.0
            rec = float(recalls[-1]) if recalls.size else 0.0
            tp = int(tps[-1]) if tps.size else 0
            fp = int(fps[-1]) if fps.size else 0
            fn = int(num_gt - tp)
            results.append(MAPResult(cat, prec, rec, ap, tp, fp, fn))
            self.logger.info(f"Category '{cat}': AP={ap:.4f} | TP={tp}, FP={fp}, FN={fn}")

        return results

    # ----------------------------
    # Process all files + save
    # ----------------------------
    def process_all_files(self) -> Tuple[List[EvaluationResult], List[MAPResult]]:
        matched = self.find_matching_files()
        if not matched:
            self.logger.error("No matched files found.")
            return [], []
        for stem, (gt_path, llm_path) in matched.items():
            res = self.evaluate_file_pair(stem, gt_path, llm_path)
            self.results.append(res)
        map_summary = self.calculate_map()
        self.map_summary = map_summary
        return self.results, map_summary

    def save_results(self):
        if not self.results:
            self.logger.warning("No evaluation results to save.")
            return

        # Similarity CSV
        sim_rows = []
        for r in self.results:
            row = {
                'filename': r.filename,
                'similarity_score': float(r.similarity_score),
                'num_gt_objects': r.num_gt_objects,
                'num_llm_detections': r.num_llm_detections,
                'num_matched_pairs': r.num_matched_pairs
            }
            for k, v in (r.individual_scores or {}).items():
                row[f"{k}_sim"] = float(v)
            sim_rows.append(row)
        sim_df = pd.DataFrame(sim_rows)
        sim_path = self.run_dir / "similarity_results.csv"
        sim_df.to_csv(sim_path, index=False)
        self.logger.info(f"Saved similarity CSV: {sim_path}")

        # Raw map_data
        map_df = pd.DataFrame(self.map_data)
        map_path = self.run_dir / "map_data.csv"
        map_df.to_csv(map_path, index=False)
        self.logger.info(f"Saved raw map_data CSV: {map_path}")

        # per-class AP
        if hasattr(self, "map_summary") and self.map_summary:
            ap_rows = [r.__dict__ for r in self.map_summary]
            ap_df = pd.DataFrame(ap_rows)
            ap_path = self.run_dir / "map_summary.csv"
            ap_df.to_csv(ap_path, index=False)
            mean_ap = ap_df['ap'].dropna().mean() if not ap_df.empty else 0.0
            self.logger.info(f"Overall mAP@{self.iou_threshold_map:.2f}: {mean_ap:.4f}")
            self.logger.info(f"Saved map summary CSV: {ap_path}")

        # plots
        try:
            if not sim_df.empty:
                plt.figure(figsize=(8, 5))
                plt.hist(sim_df['similarity_score'].fillna(0.0), bins=20, edgecolor='black', alpha=0.7)
                plt.title("Similarity Score Distribution")
                plt.xlabel("Similarity score")
                plt.ylabel("Frequency")
                plt.grid(alpha=0.4)
                hist_path = self.run_dir / "similarity_hist.png"
                plt.savefig(hist_path)
                plt.close()
                self.logger.info(f"Saved similarity histogram: {hist_path}")
        except Exception as e:
            self.logger.exception(f"Failed to plot/save histograms: {e}")

        # Save a small JSON summary
        try:
            summary = {
                'num_files': len(self.results),
                'total_gt_objects': sum(r.num_gt_objects for r in self.results),
                'total_llm_detections': sum(r.num_llm_detections for r in self.results),
                'mean_similarity': float(np.mean([r.similarity_score for r in self.results])) if self.results else 0.0
            }
            summary_path = self.run_dir / "evaluation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Saved run summary JSON: {summary_path}")
        except Exception as e:
            self.logger.exception(f"Failed to save JSON summary: {e}")
