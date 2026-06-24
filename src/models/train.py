"""
train.py
Training loop for U-Net + ResNet34 road segmentation model.
Resume-capable: saves checkpoint every epoch so training can continue
from where it left off if the session disconnects (Colab idle timeout).
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.models.unet_resnet import UNetResNet34
from src.data_pipeline.dataset import RoadDataset


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)


def train():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = config["deepglobe"]["train_dir"]
    image_size = config["model"]["input_size"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    lr = config["model"]["learning_rate"]
    val_split = config["training"]["val_split"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    best_path = os.path.join(checkpoint_dir, "best_model.pth")

    print(f"Device: {device}")

    full_dataset = RoadDataset(train_dir, image_size=image_size, augment=True, simulate_occlusion=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    model = UNetResNet34(pretrained=config["model"]["pretrained"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_val_loss = float("inf")

    # --- RESUME LOGIC: load last checkpoint if it exists ---
    if os.path.exists(resume_path):
        print(f"Found existing checkpoint at {resume_path}, resuming...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming from epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found, starting fresh.")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- SAVE CHECKPOINT EVERY EPOCH (for resume) ---
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
        }, resume_path)

        # --- SAVE BEST MODEL SEPARATELY (for inference later) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (val_loss: {val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    train()