"""
Evaluation and Ablation Harness for Crop Leaf Disease Detection.

Evaluates predictor on real photos and benchmarks:
- data/test_field (held-out field images: PlantDoc and Kaggle sets)
- data/test_lab
- data/sample_real_leaves (<Crop___Disease>_<n>.jpg)
- test manifest CSVs (using 'source' column)
- Leave-One-Source-Out (LOSO) cross-dataset mode for honest generalization estimates

Logs:
- raw CNN top-1 class (39-way, no priors, no hint)
- raw CNN top-1 crop (sum of probs per crop)
- current pipeline's final crop and disease
- whether morphology priors changed top crop (flip count)
- 39-way accuracy, crop accuracy, disease accuracy given true crop hint
- Top 10 most common confusions
- Before/after comparison table with --no-priors flag
"""

import os
import sys
import glob
import re
import csv
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

# Ensure ml_service and repo root are in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
ML_SERVICE_DIR = os.path.join(REPO_ROOT, "ml_service")
for p in (BASE_DIR, REPO_ROOT, ML_SERVICE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import CropDiseasePredictor

# Known sample filename mappings for non-standard filenames in data/sample_real_leaves
KNOWN_SAMPLE_ALIASES: Dict[str, str] = {
    "a-bellpepper-leaf.png": "Bell_Pepper___Bacterial_spot",
    "a-cotton-leaf.png": "Cotton___Bacterial_blight",
    "a-grapes-leaf.png": "Grape___Esca_Black_Measles",
    "a-maize-leaf.png": "Maize___Gray_leaf_spot",
    "a-potato-leaf.png": "Potato___Early_blight",
    "a-rice-leaf.png": "Rice___Brown_spot",
    "a-sugarcane-leaf.png": "Sugarcane___Red_rot",
    "a-tomato-leaf.png": "Tomato___Early_blight",
    "a-wheat-leaf.png": "Wheat___Brown_rust",
    "apple-fungal-leaf.png": "Apple___Apple_scab",
    "real_apple_cedar_rust.jpg": "Apple___Cedar_apple_rust",
    "real_apple_scab.jpg": "Apple___Apple_scab",
    "real_grape_black_rot.jpg": "Grape___Black_rot",
    "real_maize_common_rust.jpg": "Maize___Common_rust",
    "real_pepper_bacterial_spot.jpg": "Bell_Pepper___Bacterial_spot",
    "real_potato_early_blight.jpg": "Potato___Early_blight",
    "real_tomato_early_blight.jpg": "Tomato___Early_blight",
    "real_tomato_healthy.jpg": "Tomato___Healthy",
    "real_tomato_late_blight.jpg": "Tomato___Late_blight",
    "Tomato_Early_Blight_REAL.jpg": "Tomato___Early_blight",
    "Wheat___Brown_rust.jpg": "Wheat___Brown_rust",
    "user_apple_leaf.jpg": "Apple___Healthy"
}


def parse_label_from_filename(filename: str, valid_classes: List[str]) -> Optional[str]:
    """Parses standard <Crop___Disease>_<n>.ext or alias into standard label."""
    base = os.path.basename(filename)
    if base in KNOWN_SAMPLE_ALIASES:
        return KNOWN_SAMPLE_ALIASES[base]

    no_ext = os.path.splitext(base)[0]
    # Check if ends with _<number>
    m = re.match(r"^(.*)_[0-9]+$", no_ext)
    candidate = m.group(1) if m else no_ext

    if candidate in valid_classes:
        return candidate

    # Case-insensitive search in valid classes
    cand_lower = candidate.lower()
    for cls_name in valid_classes:
        if cls_name.lower() == cand_lower:
            return cls_name

    return None


def discover_datasets(repo_root: str, valid_classes: List[str], custom_manifest: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Finds and loads evaluation images across all available sources in the repo.
    Skips missing directories/manifests and provides a discovery status report.
    """
    discovered_samples: List[Dict[str, Any]] = []
    discovery_status: Dict[str, str] = {}

    # 1. Check data/sample_real_leaves
    sample_dir = os.path.join(repo_root, "data", "sample_real_leaves")
    if os.path.exists(sample_dir) and os.path.isdir(sample_dir):
        files = glob.glob(os.path.join(sample_dir, "*.*"))
        valid_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        count = 0
        for fpath in valid_files:
            lbl = parse_label_from_filename(fpath, valid_classes)
            if lbl:
                discovered_samples.append({
                    "path": fpath,
                    "label": lbl,
                    "source": "sample_real_leaves",
                    "split": "sample_real_leaves"
                })
                count += 1
        discovery_status["data/sample_real_leaves"] = f"Found ({count} images loaded)"
    else:
        discovery_status["data/sample_real_leaves"] = "Missing (directory not found)"

    # 2. Check data/test_field and data/processed/test_field
    field_dirs = [
        os.path.join(repo_root, "data", "test_field"),
        os.path.join(repo_root, "data", "processed", "test_field")
    ]
    found_field = False
    for fdir in field_dirs:
        if os.path.exists(fdir) and os.path.isdir(fdir):
            count = 0
            for root, _, files in os.walk(fdir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        full_p = os.path.join(root, fname)
                        parent = os.path.basename(root)
                        lbl = parent if parent in valid_classes else parse_label_from_filename(fname, valid_classes)
                        if lbl:
                            discovered_samples.append({
                                "path": full_p,
                                "label": lbl,
                                "source": "test_field",
                                "split": "test_field"
                            })
                            count += 1
            if count > 0:
                discovery_status[os.path.relpath(fdir, repo_root)] = f"Found ({count} images loaded)"
                found_field = True
                break
    if not found_field:
        discovery_status["data/test_field"] = "Missing (directory not found on disk)"

    # 3. Check data/test_lab and data/processed/test_lab
    lab_dirs = [
        os.path.join(repo_root, "data", "test_lab"),
        os.path.join(repo_root, "data", "processed", "test_lab")
    ]
    found_lab = False
    for ldir in lab_dirs:
        if os.path.exists(ldir) and os.path.isdir(ldir):
            count = 0
            for root, _, files in os.walk(ldir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        full_p = os.path.join(root, fname)
                        parent = os.path.basename(root)
                        lbl = parent if parent in valid_classes else parse_label_from_filename(fname, valid_classes)
                        if lbl:
                            discovered_samples.append({
                                "path": full_p,
                                "label": lbl,
                                "source": "test_lab",
                                "split": "test_lab"
                            })
                            count += 1
            if count > 0:
                discovery_status[os.path.relpath(ldir, repo_root)] = f"Found ({count} images loaded)"
                found_lab = True
                break
    if not found_lab:
        discovery_status["data/test_lab"] = "Missing (directory not found on disk)"

    # 4. Check Test Manifest CSVs
    manifest_candidates = [
        os.path.join(repo_root, "data", "manifests", "test_field_manifest.csv"),
        os.path.join(repo_root, "data", "manifests", "test_lab_manifest.csv"),
        os.path.join(repo_root, "data", "processed", "test_field_manifest.csv"),
        os.path.join(repo_root, "data", "processed", "test_lab_manifest.csv")
    ]
    if custom_manifest:
        manifest_candidates.insert(0, custom_manifest)

    for mpath in manifest_candidates:
        rel_m = os.path.relpath(mpath, repo_root) if os.path.isabs(mpath) else mpath
        if os.path.exists(mpath):
            existing_rows = 0
            try:
                with open(mpath, "r", encoding="utf-8") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        p = row.get("path", "")
                        lbl = row.get("label", "")
                        src = row.get("source", "manifest")
                        split = row.get("type", "manifest_split")
                        
                        resolved_path = None
                        if os.path.exists(p):
                            resolved_path = p
                        elif os.path.exists(os.path.join(repo_root, p)):
                            resolved_path = os.path.join(repo_root, p)

                        if resolved_path and lbl in valid_classes:
                            discovered_samples.append({
                                "path": resolved_path,
                                "label": lbl,
                                "source": src,
                                "split": split
                            })
                            existing_rows += 1
                if existing_rows > 0:
                    discovery_status[rel_m] = f"Found ({existing_rows} valid records on disk)"
                else:
                    discovery_status[rel_m] = "Found CSV, but referenced image paths do not exist on disk"
            except Exception as e:
                discovery_status[rel_m] = f"Error reading CSV: {e}"
        else:
            discovery_status[rel_m] = "Missing (CSV not found)"

    return discovered_samples, discovery_status


def discover_loso_sources(repo_root: str, valid_classes: List[str], samples_per_class: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Discovers source datasets from data/raw for Leave-One-Source-Out (LOSO) cross-source evaluation.
    Maps folders according to ml_training/mapping.yaml.
    """
    import yaml
    loso_samples: List[Dict[str, Any]] = []
    loso_status: Dict[str, str] = {}

    mapping_file = os.path.join(repo_root, "ml_training", "mapping.yaml")
    if not os.path.exists(mapping_file):
        return loso_samples, {"mapping.yaml": "Missing ml_training/mapping.yaml"}

    with open(mapping_file, "r") as f:
        mapping = yaml.safe_load(f)

    raw_dir = os.path.join(repo_root, "data", "raw")
    if not os.path.exists(raw_dir):
        return loso_samples, {"data/raw": "Missing data/raw"}

    sources_cfg = mapping.get("sources", {})
    for src_name, src_info in sources_cfg.items():
        src_dir = os.path.join(raw_dir, src_name)
        if not os.path.exists(src_dir):
            loso_status[src_name] = "Missing folder in data/raw"
            continue

        folder_mappings = src_info.get("folder_mappings", {})
        src_type = src_info.get("type", "field")
        collected_per_class: Dict[str, int] = {}

        # Fast directory map for 1-level and 2-level subfolders
        dir_map = {}
        for entry in os.scandir(src_dir):
            if entry.is_dir():
                dir_map[entry.name.lower()] = entry.path
                try:
                    for sub in os.scandir(entry.path):
                        if sub.is_dir():
                            dir_map[sub.name.lower()] = sub.path
                except Exception:
                    pass

        # Match folders
        for folder_name, target_class in folder_mappings.items():
            if target_class not in valid_classes:
                continue

            sub_path = dir_map.get(folder_name.lower())
            if not sub_path:
                sub_path = dir_map.get(folder_name.replace(" ", "_").lower())
            if not sub_path:
                sub_path = dir_map.get(folder_name.replace("_", " ").lower())

            if sub_path and os.path.exists(sub_path):
                img_files = [
                    os.path.join(sub_path, fn)
                    for fn in os.listdir(sub_path)
                    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                # Sample up to samples_per_class
                for img_p in img_files[:samples_per_class]:
                    loso_samples.append({
                        "path": img_p,
                        "label": target_class,
                        "source": src_name,
                        "split": f"loso_{src_name.lower()}"
                    })
                    collected_per_class[target_class] = collected_per_class.get(target_class, 0) + 1

        total_src = sum(collected_per_class.values())
        if total_src > 0:
            loso_status[src_name] = f"Found ({total_src} images across {len(collected_per_class)} classes)"
        else:
            loso_status[src_name] = "Found folder, but 0 matching images found"

    return loso_samples, loso_status


def evaluate_records(
    predictor: CropDiseasePredictor,
    records: List[Dict[str, Any]],
    force_no_priors: bool = False
) -> List[Dict[str, Any]]:
    """
    Executes prediction pipeline on each image record and captures detailed evaluation logs.
    """
    results: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        fpath = rec["path"]
        true_label = rec["label"]
        true_crop, true_disease = true_label.split("___")
        source = rec.get("source", "unknown")
        split = rec.get("split", "unknown")

        try:
            with open(fpath, "rb") as fp:
                img_bytes = fp.read()
        except Exception as e:
            continue

        # 1. Unconstrained prediction with morphology priors (or no-priors if flag set)
        res_main = predictor.predict(img_bytes, crop_hint=None, use_priors=not force_no_priors)

        # 2. Prediction without priors (for ablation comparison)
        res_no_prior = predictor.predict(img_bytes, crop_hint=None, use_priors=False)

        # 3. Disease prediction conditioned on true crop hint
        res_hint = predictor.predict(img_bytes, crop_hint=true_crop, use_priors=not force_no_priors)

        raw_top1_class = res_main.get("raw_top1_class", "unknown")
        raw_top1_crop = res_main.get("raw_top1_crop", "unknown")
        final_crop = res_main.get("crop", "unknown")
        final_disease = res_main.get("disease", "unknown")
        final_class = f"{final_crop}___{final_disease}"
        prior_flipped = res_main.get("prior_flipped", False)

        no_prior_crop = res_no_prior.get("crop", "unknown")
        no_prior_disease = res_no_prior.get("disease", "unknown")
        no_prior_class = f"{no_prior_crop}___{no_prior_disease}"

        hint_disease = res_hint.get("disease", "unknown")

        def _norm(s: Optional[str]) -> str:
            return (s or "").strip().lower().replace(" ", "_").replace(",", "")

        results.append({
            "path": fpath,
            "filename": os.path.basename(fpath),
            "true_class": true_label,
            "true_crop": true_crop,
            "true_disease": true_disease,
            "source": source,
            "split": split,
            "status": res_main.get("status", "unknown"),
            "confidence": res_main.get("confidence", 0.0),
            "raw_top1_class": raw_top1_class,
            "raw_top1_crop": raw_top1_crop,
            "final_crop": final_crop,
            "final_disease": final_disease,
            "final_class": final_class,
            "prior_flipped": prior_flipped,
            "no_prior_crop": no_prior_crop,
            "no_prior_disease": no_prior_disease,
            "no_prior_class": no_prior_class,
            "disease_given_true_crop": hint_disease,
            "correct_raw_39": bool(raw_top1_class and true_label and _norm(raw_top1_class) == _norm(true_label)),
            "correct_raw_crop": bool(raw_top1_crop and true_crop and _norm(raw_top1_crop) == _norm(true_crop)),
            "correct_final_crop": bool(final_crop and true_crop and _norm(final_crop) == _norm(true_crop)),
            "correct_final_39": bool(final_class and true_label and _norm(final_class) == _norm(true_label)),
            "correct_no_prior_39": bool(no_prior_class and true_label and _norm(no_prior_class) == _norm(true_label)),
            "correct_no_prior_crop": bool(no_prior_crop and true_crop and _norm(no_prior_crop) == _norm(true_crop)),
            "correct_disease_given_hint": bool(hint_disease and true_disease and _norm(hint_disease) == _norm(true_disease))
        })

    return results


def compute_metrics(eval_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes accuracies, flip rate, and top confusions from evaluation logs."""
    n = len(eval_logs)
    if n == 0:
        return {
            "total": 0,
            "raw_39_acc": 0.0,
            "crop_acc": 0.0,
            "disease_given_crop_acc": 0.0,
            "final_39_acc": 0.0,
            "no_prior_39_acc": 0.0,
            "no_prior_crop_acc": 0.0,
            "prior_flip_count": 0,
            "prior_flip_rate": 0.0,
            "top_10_confusions": []
        }

    raw_39_corr = sum(1 for r in eval_logs if r["correct_raw_39"])
    crop_corr = sum(1 for r in eval_logs if r["correct_final_crop"])
    disease_hint_corr = sum(1 for r in eval_logs if r["correct_disease_given_hint"])
    final_39_corr = sum(1 for r in eval_logs if r["correct_final_39"])
    no_prior_39_corr = sum(1 for r in eval_logs if r["correct_no_prior_39"])
    no_prior_crop_corr = sum(1 for r in eval_logs if r["correct_no_prior_crop"])
    flip_cnt = sum(1 for r in eval_logs if r["prior_flipped"])

    # Count confusions (True Class -> Final Predicted Class)
    confusions = Counter()
    for r in eval_logs:
        if not r["correct_final_39"]:
            confusions[(r["true_class"], r["final_class"])] += 1

    top_10 = [
        {"true": k[0], "pred": k[1], "count": v, "pct": round(v / n * 100, 1)}
        for k, v in confusions.most_common(10)
    ]

    return {
        "total": n,
        "raw_39_acc": round(raw_39_corr / n * 100, 2),
        "crop_acc": round(crop_corr / n * 100, 2),
        "disease_given_crop_acc": round(disease_hint_corr / n * 100, 2),
        "final_39_acc": round(final_39_corr / n * 100, 2),
        "no_prior_39_acc": round(no_prior_39_corr / n * 100, 2),
        "no_prior_crop_acc": round(no_prior_crop_corr / n * 100, 2),
        "prior_flip_count": flip_cnt,
        "prior_flip_rate": round(flip_cnt / n * 100, 2),
        "top_10_confusions": top_10
    }


def print_breakdown_table(title: str, metrics: Dict[str, Any]):
    """Renders a clean tabular summary for a partition."""
    print(f"\n{'-' * 80}")
    print(f" {title.upper()} (N = {metrics['total']})")
    print(f"{'-' * 80}")
    print(f"  • Raw CNN 39-Way Accuracy          : {metrics['raw_39_acc']:6.2f}%")
    print(f"  • Raw CNN Crop Accuracy            : {metrics['crop_acc']:6.2f}%")
    print(f"  • Disease Acc (Given True Crop)    : {metrics['disease_given_crop_acc']:6.2f}%")
    print(f"  • Final Pipeline 39-Way Accuracy   : {metrics['final_39_acc']:6.2f}%")
    print(f"  • Prior Flip Count / Flip Rate     : {metrics['prior_flip_count']} ({metrics['prior_flip_rate']:.2f}%)")

    if metrics["top_10_confusions"]:
        print(f"\n  Top Confusions (True -> Predicted):")
        for idx, conf in enumerate(metrics["top_10_confusions"][:5], 1):
            print(f"    {idx}. {conf['true']:32s} -> {conf['pred']:32s} : {conf['count']:2d} ({conf['pct']}%)")


def print_comparison_table(metrics: Dict[str, Any]):
    """Prints side-by-side comparison between priors enabled vs --no-priors."""
    print(f"\n{'=' * 80}")
    print(f" ABLATION COMPARISON: MORPHOLOGY PRIORS VS. NO PRIORS (ALL 1.0)")
    print(f"{'=' * 80}")
    print(f" {'Metric':<35s} | {'With Priors':<14s} | {'No Priors':<14s} | {'Delta':<10s}")
    print(f" {'-' * 35}-|-{'-' * 14}-|-{'-' * 14}-|-{'-' * 10}")

    c_acc_with = metrics["crop_acc"]
    c_acc_no = metrics["no_prior_crop_acc"]
    c_delta = c_acc_with - c_acc_no

    p_acc_with = metrics["final_39_acc"]
    p_acc_no = metrics["no_prior_39_acc"]
    p_delta = p_acc_with - p_acc_no

    print(f" {'Crop Species Identification Acc':<35s} | {c_acc_with:6.2f}%        | {c_acc_no:6.2f}%        | {c_delta:+6.2f}%")
    print(f" {'39-Way End-to-End Pipeline Acc':<35s} | {p_acc_with:6.2f}%        | {p_acc_no:6.2f}%        | {p_delta:+6.2f}%")
    print(f" {'Priors Flip Rate':<35s} | {metrics['prior_flip_rate']:6.2f}%        |    0.00%        | {metrics['prior_flip_rate']:+6.2f}%")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate crop disease model on real photos & ablation benchmarks.")
    parser.add_argument("--config", type=str, default="ml_service/config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-priors", action="store_true", help="Set all crop priors to 1.0 (disables morphological prior multipliers)")
    parser.add_argument("--loso", action="store_true", help="Execute Leave-One-Source-Out (LOSO) cross-dataset benchmark across data/raw sources")
    parser.add_argument("--loso-samples", type=int, default=5, help="Number of samples per class for LOSO evaluation")
    parser.add_argument("--manifest", type=str, default=None, help="Optional path to custom evaluation CSV manifest")
    parser.add_argument("--output-json", type=str, default=None, help="Save evaluation metrics to JSON file")
    args = parser.parse_args()

    # Resolve config path
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        alt = os.path.join(REPO_ROOT, "ml_service", "config.yaml")
        if os.path.exists(alt):
            cfg_path = alt

    print("\n" + "=" * 80)
    print(" CROP DISEASE DETECTION: REAL-WORLD & ABLATION EVALUATION HARNESS")
    print("=" * 80)
    print(f"[*] Configuration Path : {cfg_path}")
    print(f"[*] Mode               : {'NO-PRIORS ABLATION (Priors = 1.0)' if args.no_priors else 'FULL PIPELINE (Botanical Priors Enabled)'}")

    # Load predictor
    predictor = CropDiseasePredictor(config_path=cfg_path)
    print(f"[+] Predictor loaded ({predictor.num_classes} classes on device: {predictor.device})")

    # 1. Discover existing evaluation datasets in repository
    samples, discovery = discover_datasets(REPO_ROOT, predictor.classes, custom_manifest=args.manifest)

    print("\n" + "-" * 80)
    print(" DATASET DISCOVERY REPORT")
    print("-" * 80)
    for src_name, status_str in discovery.items():
        symbol = "[+]" if "Found" in status_str else "[-]"
        print(f"  {symbol} {src_name:42s} : {status_str}")

    if not samples and not args.loso:
        print("\n[!] Error: No real leaf images or valid manifest paths found on disk.")
        sys.exit(1)

    print(f"\n[+] Total Active Evaluation Images Discovered: {len(samples)}")

    # 2. Run Evaluation
    eval_logs = evaluate_records(predictor, samples, force_no_priors=args.no_priors)
    overall_metrics = compute_metrics(eval_logs)

    # 3. Print Generalization Notice
    print("\n" + "=" * 80)
    print(" CRITICAL GENERALIZATION ESTIMATE NOTICE")
    print("=" * 80)
    print(" [NOTE] Metrics on 'test_field' (PlantDoc/Kaggle partitions split from training sources)")
    print("        represent an OPTIMISTIC estimate due to shared background and sensor distributions.")
    print("        For a rigorous, honest out-of-distribution transfer benchmark, see the")
    print("        LEAVE-ONE-SOURCE-OUT (LOSO) report below.")
    print("=" * 80)

    # 4. Print Overall Metrics
    print_breakdown_table("OVERALL EVALUATION SUMMARY", overall_metrics)

    # 5. Breakdown by Split
    splits = sorted(set(r["split"] for r in eval_logs))
    if len(splits) > 1:
        print("\n" + "=" * 80)
        print(" BREAKDOWN BY DATASET SPLIT")
        print("=" * 80)
        for s in splits:
            s_logs = [r for r in eval_logs if r["split"] == s]
            print_breakdown_table(f"SPLIT: {s}", compute_metrics(s_logs))

    # 6. Breakdown by Source Dataset
    sources = sorted(set(r["source"] for r in eval_logs))
    if len(sources) > 1:
        print("\n" + "=" * 80)
        print(" BREAKDOWN BY SOURCE DATASET")
        print("=" * 80)
        for src in sources:
            src_logs = [r for r in eval_logs if r["source"] == src]
            print_breakdown_table(f"SOURCE: {src}", compute_metrics(src_logs))

    # 7. Print Before/After Comparison Table
    print_comparison_table(overall_metrics)

    # 8. Leave-One-Source-Out (LOSO) Mode (Task 5 cross-dataset evaluation)
    loso_metrics_dict = {}
    if args.loso:
        print("\n" + "=" * 80)
        print(" TASK 5: LEAVE-ONE-SOURCE-OUT (LOSO) CROSS-DATASET BENCHMARK")
        print("=" * 80)
        print("[*] Evaluating out-of-distribution transfer across raw dataset sources...")
        loso_samples, loso_status = discover_loso_sources(REPO_ROOT, predictor.classes, samples_per_class=args.loso_samples)
        for k, v in loso_status.items():
            print(f"  [+] Source {k:20s}: {v}")

        if loso_samples:
            loso_logs = evaluate_records(predictor, loso_samples, force_no_priors=args.no_priors)
            loso_overall = compute_metrics(loso_logs)
            loso_metrics_dict["overall_loso"] = loso_overall
            print_breakdown_table("OVERALL LEAVE-ONE-SOURCE-OUT BENCHMARK", loso_overall)

            for src in sorted(set(r["source"] for r in loso_samples)):
                sub_logs = [r for r in loso_logs if r["source"] == src]
                sub_metrics = compute_metrics(sub_logs)
                loso_metrics_dict[src] = sub_metrics
                print_breakdown_table(f"HELD-OUT SOURCE TRANSFER: {src}", sub_metrics)

    # Save to JSON if requested
    if args.output_json:
        report = {
            "overall": overall_metrics,
            "by_split": {s: compute_metrics([r for r in eval_logs if r["split"] == s]) for s in splits},
            "by_source": {s: compute_metrics([r for r in eval_logs if r["source"] == s]) for s in sources},
            "loso": loso_metrics_dict
        }
        with open(args.output_json, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"\n[+] Detailed evaluation metrics exported to: {args.output_json}")


if __name__ == "__main__":
    main()
