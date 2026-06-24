"""
download.py
Verifies that the DeepGlobe dataset is correctly downloaded and paired.
Run this once after downloading data to confirm everything is in order.
"""

import os
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_dataset(train_dir: str):
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    files = os.listdir(train_dir)
    sat_files = sorted([f for f in files if f.endswith("_sat.jpg")])
    mask_files = sorted([f for f in files if f.endswith("_mask.png")])

    sat_ids = {f.replace("_sat.jpg", "") for f in sat_files}
    mask_ids = {f.replace("_mask.png", "") for f in mask_files}

    missing_masks = sat_ids - mask_ids
    missing_sats = mask_ids - sat_ids

    print(f"Satellite images found: {len(sat_files)}")
    print(f"Mask images found: {len(mask_files)}")
    print(f"Matched pairs: {len(sat_ids & mask_ids)}")

    if missing_masks:
        print(f"WARNING: {len(missing_masks)} satellite images have no mask")
    if missing_sats:
        print(f"WARNING: {len(missing_sats)} masks have no satellite image")

    if not missing_masks and not missing_sats:
        print("Dataset verification PASSED. All pairs match.")

    return len(sat_ids & mask_ids)


if __name__ == "__main__":
    config = load_config()
    train_dir = config["deepglobe"]["train_dir"]
    verify_dataset(train_dir)