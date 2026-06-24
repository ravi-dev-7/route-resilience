"""
dataset.py
PyTorch Dataset class for loading DeepGlobe satellite images and road masks.
Includes synthetic occlusion simulation (canopy + shadow patches) so the
model learns to infer road continuity under real-world occlusion, as
specified by ISRO's problem statement.
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def apply_synthetic_occlusion(image: np.ndarray, num_patches: int = 3, max_patch_frac: float = 0.18) -> np.ndarray:
    """
    Simulates real-world occlusion (tree canopy + building shadows) by
    overlaying random patches on the satellite image. The mask is left
    UNCHANGED - the model must learn to predict the true road location
    even though it's visually hidden, which is the core challenge ISRO
    describes ("occlusion-robust segmentation").

    Args:
        image: RGB satellite image (H, W, 3)
        num_patches: how many occlusion patches to apply per image
        max_patch_frac: max patch size as a fraction of image dimension

    Returns:
        Occluded image (same shape, mask is untouched elsewhere)
    """
    occluded = image.copy()
    h, w = image.shape[:2]

    for _ in range(num_patches):
        patch_type = random.choice(["canopy", "shadow"])

        patch_h = random.randint(int(h * 0.05), int(h * max_patch_frac))
        patch_w = random.randint(int(w * 0.05), int(w * max_patch_frac))

        y = random.randint(0, max(h - patch_h, 1))
        x = random.randint(0, max(w - patch_w, 1))

        region = occluded[y:y + patch_h, x:x + patch_w]

        if patch_type == "canopy":
            # green-tinted, semi-transparent overlay simulating tree cover
            overlay = np.zeros_like(region)
            overlay[:, :, 1] = 90   # boost green channel
            blended = cv2.addWeighted(region, 0.4, overlay, 0.6, 0)
            occluded[y:y + patch_h, x:x + patch_w] = blended
        else:
            # dark, hard-edged overlay simulating building/cloud shadow
            darkened = (region * 0.25).astype(np.uint8)
            occluded[y:y + patch_h, x:x + patch_w] = darkened

    return occluded


class RoadDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 512, augment: bool = True,
                 simulate_occlusion: bool = True, occlusion_prob: float = 0.5):
        self.image_dir = image_dir
        self.image_size = image_size
        self.simulate_occlusion = simulate_occlusion
        self.occlusion_prob = occlusion_prob

        files = os.listdir(image_dir)
        self.ids = sorted({f.replace("_sat.jpg", "") for f in files if f.endswith("_sat.jpg")})

        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.image_dir, f"{img_id}_sat.jpg")
        mask_path = os.path.join(self.image_dir, f"{img_id}_mask.png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # binarize: road = 1, background = 0

        # apply synthetic occlusion to the IMAGE ONLY - mask stays ground truth
        if self.simulate_occlusion and random.random() < self.occlusion_prob:
            image = apply_synthetic_occlusion(image)

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"].unsqueeze(0)  # add channel dim

        return image_tensor, mask_tensor


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["deepglobe"]["train_dir"]
    size = config["model"]["input_size"]

    dataset = RoadDataset(train_dir, image_size=size, simulate_occlusion=True, occlusion_prob=1.0)
    print(f"Total samples: {len(dataset)}")

    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print("Occlusion simulation active: roads occluded in image, mask remains ground-truth.")