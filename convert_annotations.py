import os, json, glob
import numpy as np
import re
import h5py
from tqdm import tqdm

CLASS_MAP = {"PO": 0, "PWP": 1, "PWPK": 2, "PWK": 3}

def find_radar_dir(session_dir):
    """Find radar ADC folder (usually starts with 'Cascade_Capture_')."""
    for d in os.listdir(session_dir):
        full = os.path.join(session_dir, d)
        if os.path.isdir(full) and d.lower().startswith("cascade_capture"):
            return full
    return None


def convert_annotations(session_dir):
    ann_path = os.path.join(session_dir, "2D-on-2D_annotations_export.json")
    radar_dir = find_radar_dir(session_dir)
    label_out = os.path.join(session_dir, "labels_json")
    os.makedirs(label_out, exist_ok=True)

    if radar_dir is None:
        print(f"⚠️ No radar folder found in {session_dir}")
        return
    if not os.path.exists(ann_path):
        print(f"⚠️ No 2D-on-2D_annotations_export.json found in {session_dir}")
        return

    # Load color-frame annotations
    with open(ann_path, "r") as f:
        ann_data = json.load(f).get("images", [])
        #print(ann_data)


    # Sort by the number in the filename
    ann_data_sorted = sorted(
        ann_data,
        key=lambda x: int(re.search(r'(\d+)', x["image"]).group())
    )

    #print(ann_data_sorted)


    # Get all radar .mat files in sorted order
    radar_files = sorted(glob.glob(os.path.join(radar_dir, "*.mat")))
    #print(radar_files)

    if len(radar_files) != len(ann_data_sorted):
        print(f"⚠️ Frame count mismatch: {len(radar_files)} radar vs {len(ann_data_sorted)} annotations in {session_dir}")

    n = min(len(radar_files), len(ann_data_sorted))
    for i in range(n):
        radar_file = radar_files[i]
        entry = ann_data_sorted[i]

        boxes = []
        for ann in entry.get("annotations", []):
            cls_name = ann.get("class")
            if cls_name not in CLASS_MAP:
                continue
            cid = CLASS_MAP[cls_name]
            bb = ann["boundingBox"]
            x1, y1 = bb["x"], bb["y"]
            x2, y2 = x1 + bb["width"], y1 + bb["height"]
            boxes.append({"class_id": cid, "bbox_xyxy": [x1, y1, x2, y2]})

        # Write one JSON per radar frame
        base = os.path.splitext(os.path.basename(radar_file))[0]
        out_path = os.path.join(label_out, f"{base}.json")
        with open(out_path, "w") as f:
            json.dump({"boxes": boxes}, f, indent=2)

    print(f"✅ Converted {n} paired frames for {os.path.basename(session_dir)}")


def convert_all(root_dir):
    """Recursively convert all people and sessions under the root."""
    for person in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue
        for sess in sorted(os.listdir(person_path)):
            sess_path = os.path.join(person_path, sess)
            if os.path.isdir(sess_path):
                convert_annotations(sess_path)




def adc_to_ear(adc):
    """
    Convert raw ADC data [Tx, Rx, Chirps, Samples] -> EAR [E, A, R]
    Performs FFTs over range, azimuth (Rx), and elevation (Tx).
    """
    # 1. Range FFT along samples
    range_fft = np.fft.fft(adc, axis=-1)
    range_fft = range_fft[..., :adc.shape[-1]]  # keep positive half

    # 2. Average across chirps (collapse Doppler)
    mean_chirp = np.mean(range_fft, axis=2)  # [Tx, Rx, Range]

    # 3. Elevation FFT across Tx antennas
    elev_fft = np.fft.fftshift(np.fft.fft(mean_chirp, axis=0), axes=0)

    # 4. Azimuth FFT across Rx antennas
    azim_fft = np.fft.fftshift(np.fft.fft(elev_fft, axis=1), axes=1)

    # 5. Magnitude and normalization
    EAR = np.abs(azim_fft)
    EAR = EAR / (np.max(EAR) + 1e-6)  # normalize to [0,1]
    return EAR.astype(np.float32)

def adc_to_ra(adc):
    """
    Convert raw ADC data [Tx, Rx, Chirps, Samples] -> RA [R, A]
    Radar-native for RODNet.
    """
    # 1. Range FFT
    range_fft = np.fft.fft(adc, axis=-1)
    range_fft = range_fft[..., :adc.shape[-1]]

    # 2. Average across chirps (collapse Doppler)
    mean_chirp = np.mean(range_fft, axis=2)  # [Tx, Rx, R]

    # 3. Azimuth FFT across Rx antennas
    azim_fft = np.fft.fftshift(np.fft.fft(mean_chirp, axis=1), axes=1)

    # 4. Collapse Tx (sum or mean) → remove elevation
    ra = np.mean(azim_fft, axis=0)   # [RxFFT, R]

    # 5. Magnitude, transpose to (R, A)
    RA = np.abs(ra).T
    RA = RA / (np.max(RA) + 1e-6)

    return RA.astype(np.float32)


# def convert_mat_to_ear_npy(radar_dir):
#     """
#     Converts .mat radar ADC frames (adcData) → EAR numpy cubes.
#     Saves them in radar_dir/ear_frames/
#     """
#     ear_dir = os.path.join(radar_dir, "ear_frames")
#     os.makedirs(ear_dir, exist_ok=True)

#     mat_files = sorted([f for f in os.listdir(radar_dir) if f.endswith(".mat")])
#     if not mat_files:
#         print(f"⚠️ No .mat files found in {radar_dir}")
#         return

#     for mf in tqdm(mat_files, desc=f"Processing {os.path.basename(radar_dir)}"):
#         mat_path = os.path.join(radar_dir, mf)
#         try:
#             with h5py.File(mat_path, "r") as f:
#                 dset = f["adcData"]
#                 adc = np.array(dset)  # (Tx, Rx, Chirps, Samples)

#             # Handle MATLAB complex struct (optional, if your data is complex split)
#             if hasattr(adc, "dtype") and adc.dtype.names == ("real", "imag"):
#                 adc = adc["real"] + 1j * adc["imag"]

#             ear = adc_to_ear(adc)
#             base = os.path.splitext(mf)[0]
#             np.save(os.path.join(ear_dir, f"{base}.npy"), ear)

#         except Exception as e:
#             print(f"❌ Error in {mf}: {e}")

#     print(f"✅ Saved {len(mat_files)} EAR .npy cubes in {ear_dir}")
#     return ear_dir

def convert_mat_to_ear_ra_npy(radar_dir):
    ear_dir = os.path.join(radar_dir, "ear_frames")
    ra_dir  = os.path.join(radar_dir, "ra_frames")
    os.makedirs(ear_dir, exist_ok=True)
    os.makedirs(ra_dir, exist_ok=True)

    mat_files = sorted([f for f in os.listdir(radar_dir) if f.endswith(".mat")])

    for mf in tqdm(mat_files, desc=f"Processing {os.path.basename(radar_dir)}"):
        mat_path = os.path.join(radar_dir, mf)
        with h5py.File(mat_path, "r") as f:
            adc = np.array(f["adcData"])

        if hasattr(adc, "dtype") and adc.dtype.names == ("real", "imag"):
            adc = adc["real"] + 1j * adc["imag"]

        ear = adc_to_ear(adc)
        ra  = adc_to_ra(adc)

        base = os.path.splitext(mf)[0]
        np.save(os.path.join(ear_dir, f"{base}.npy"), ear)
        np.save(os.path.join(ra_dir,  f"{base}.npy"), ra)

    print(f"✅ Saved EAR + RA frames in {radar_dir}")


def find_radar_dirs(root_dir):
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d.lower().startswith("cascade_capture"):
                yield os.path.join(root, d)


def convert_all_to_ear(root_dir):
    """
    Recursively converts all Cascade_Capture_* radar folders in root_dir to EAR format.
    """
    for radar_dir in find_radar_dirs(root_dir):
        convert_mat_to_ear_ra_npy(radar_dir)

    print(f"🎯 Done converting all radar sessions to EAR format under {root_dir}")



if __name__ == "__main__":
    root = r"C:\Users\n1071552\Desktop\projects\data_collectn\test_data"
    convert_all(root)
    convert_all_to_ear(root)
