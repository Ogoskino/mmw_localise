import os
import json
import random
import argparse
from config import load_config

def build_items_multi(root_dir, image_exts=(".png", ".jpg", ".jpeg")):
    items = []

    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == "ear_frames":
            ear_dir = root
            cascade_dir = os.path.dirname(ear_dir)
            session_dir = os.path.dirname(cascade_dir)

            label_dir = os.path.join(session_dir, "labels_json")
            color_dir = os.path.join(session_dir, "color_frames_")

            if not os.path.isdir(label_dir):
                print(f"⚠️ Skipping {session_dir} – missing labels_json/")
                continue
            if not os.path.isdir(color_dir):
                print(f"⚠️ Skipping {session_dir} – missing color_frames_/")
                continue

            ear_files = sorted([f for f in os.listdir(ear_dir) if f.endswith(".npy")])
            lab_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
            img_files = sorted([
                f for f in os.listdir(color_dir)
                if f.lower().endswith(image_exts)
            ])

            n = min(len(ear_files), len(lab_files), len(img_files))
            if n == 0:
                print(f"⚠️ Session {session_dir}: no valid triples")
                continue

            print(f"➡️ Session found: {session_dir} | pairing {n} frames")

            for i in range(n):
                ear_path = os.path.join(ear_dir, ear_files[i])
                lab_path = os.path.join(label_dir, lab_files[i])
                img_path = os.path.join(color_dir, img_files[i])
                items.append((ear_path, lab_path, img_path))

    print(f"✅ Total triples found: {len(items)}")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    root_dir = config["paths"]["train_root"]
    split_json = config["paths"]["split_json"]
    seed = config["split"]["seed"]
    val_size = config["split"]["val_size"]

    items = build_items_multi(root_dir)
    n = len(items)

    if n == 0:
        raise RuntimeError(f"No samples found under {root_dir}")

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # 👇 KEEP SUBJECT SPLIT
    split_dict = {
        "train": indices[val_size:],
        "val": indices[:val_size]
    }

    with open(split_json, "w") as f:
        json.dump(split_dict, f, indent=2)

    print(
        f"\n✅ Saved {split_json} with "
        f"{len(split_dict['train'])} train and {len(split_dict['val'])} val samples.\n"
    )