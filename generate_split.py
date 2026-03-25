### generate_split.py
import os
import json
import random

# =========================================================
#  Folder traversal: build (ear, label, color) triples
#  (must match YOLO & FRCNN versions exactly)
# =========================================================

def build_items_multi(root_dir, image_exts=(".png", ".jpg", ".jpeg")):
    """
    Structure:

        P_X/
          P_X_S_Y/
            labels_json/
            color_frames_/
            Cascade_Capture_pXX_sYY/
                ear_frames/

    We detect 'ear_frames', then go UP two levels to find labels_json
    and color_frames_ and pair files by sorted order.
    """
    items = []

    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == "ear_frames":
            ear_dir = root
            cascade_dir = os.path.dirname(ear_dir)        # .../Cascade_Capture_pXX_sYY
            session_dir = os.path.dirname(cascade_dir)    # .../P_X_S_Y

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
    # TODO: adjust this if your dataset root moves
    root_dir = r"C:\Users\n1071552\Desktop\projects\data_collectn\realsense_data_OD"

    items = build_items_multi(root_dir)
    n = len(items)
    if n == 0:
        raise RuntimeError(f"No samples found under {root_dir}")

    indices = list(range(n))
    random.seed(42)  # FIXED for reproducibility
    random.shuffle(indices)

    split = int(0.80 * n)
    split_dict = {
        "train": indices[1062:],
        "val":   indices[:1062]
    }

    with open("dataset_split.json", "w") as f:
        json.dump(split_dict, f, indent=2)

    print(f"\n✅ Saved dataset_split.json with {len(split_dict['train'])} train and {len(split_dict['val'])} val samples.\n")
