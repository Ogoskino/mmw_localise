"""
TRAINING SCRIPT FOR YOLO MODELS USING MMWAVE RADAR DATA
Uses yolo_utils.py for all utility functions.
"""

import json
import argparse
from config import load_config
from ultralytics import YOLO

from yolo_utils import (
    build_items_multi,
    export_yolo_dataset,
    find_best_operating_point,
    visualize_predictions,
)

# =========================================================
# CONFIGURATION
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

config = load_config(args.config)

# Training settings
W = config["training"]["img_width"]
H = config["training"]["img_height"]
EPOCHS = config["training"]["epochs"]
BATCH = config["training"]["batch_size"]
DEVICE = config["training"].get("device", 0)

# Paths
ROOT_DIR = config["paths"]["train_root"]
SPLIT_JSON = config["paths"]["split_json"]
YOLO_DS_OUT = config["paths"]["yolo_dataset_out"]
OUT_PROJECT = config["paths"]["yolo_project_out"]
BEST_VIS_OUT = config["paths"]["best_vis_out"]

# Models
YOLO_WEIGHTS = config["models"]
MODEL_NAMES = list(YOLO_WEIGHTS.keys())
PRETRAIN_WEIGHTS = list(YOLO_WEIGHTS.values())


# =========================================================
# TRAIN PIPELINE
# =========================================================

def train_yolo(root_dir):
    # -----------------------------------------------------
    # 1. Build training + validation datasets
    # -----------------------------------------------------
    print("\nScanning dataset...")
    items = build_items_multi(root_dir)

    if len(items) == 0:
        raise RuntimeError("❌ ERROR: No samples found in dataset directory!")

    with open(SPLIT_JSON, "r") as f:
        split_data = json.load(f)

    train_items = [items[i] for i in split_data["train"]]
    val_items = [items[i] for i in split_data["val"]]

    print(f"\nUsing unified split → {len(train_items)} train, {len(val_items)} val samples.")

    yaml_path, val_paths = export_yolo_dataset(
        train_items,
        val_items,
        out_root=YOLO_DS_OUT,
        W=W,
        H=H
    )

    # -----------------------------------------------------
    # 2. Train each YOLO model
    # -----------------------------------------------------
    model_summaries = []

    for name, pretrained in zip(MODEL_NAMES, PRETRAIN_WEIGHTS):
        print("\n====================================================")
        print(f" TRAINING MODEL → {name}")
        print("====================================================\n")

        model = YOLO(pretrained)

        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=H,
            batch=BATCH,
            device=DEVICE,
            project=OUT_PROJECT,
            name=name,
            exist_ok=True
        )

        # -------------------------------------------------
        # 3. Evaluate → find best confidence threshold
        # -------------------------------------------------
        print("\nSearching for best operating point...")
        best = find_best_operating_point(model, val_items, val_paths)

        best_conf = best["best_conf"]
        best_metrics = best["best_metrics"]

        # -------------------------------------------------
        # 4. Visualise predictions
        # -------------------------------------------------
        visualize_predictions(
            model,
            val_items,
            val_paths,
            conf=best_conf,
            out_dir=f"{BEST_VIS_OUT}/{name}",
            K=10
        )

        # -------------------------------------------------
        # 5. Store comparison row
        # -------------------------------------------------
        model_summaries.append({
            "model": name,
            "conf": best_conf,
            "score": best["best_score"],
            "MR": best_metrics["MR"],
            "FAR": best_metrics["FAR"],
            "F1": best_metrics["f1"]
        })

    # -----------------------------------------------------
    # 6. Print comparison table
    # -----------------------------------------------------
    print("\n\n================ MODEL COMPARISON ==================\n")
    header = f"{'Model':<12} {'Conf':<6} {'Score':<12} {'MR':<8} {'FAR':<8} {'F1':<8}"
    print(header)
    print("-" * len(header))

    for row in model_summaries:
        print(
            f"{row['model']:<12}"
            f"{row['conf']:<6.2f}"
            f"{row['score']:<12.4f}"
            f"{row['MR']:<8.3f}"
            f"{row['FAR']:<8.3f}"
            f"{row['F1']:<8.3f}"
        )

    print("\n====================================================")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    train_yolo(ROOT_DIR)