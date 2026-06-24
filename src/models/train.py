"""
train.py
Training loop for U-Net + ResNet34 road segmentation model.
Runs on CPU locally for testing, switch device in config.yaml for Colab GPU.
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
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() or config["model"]["device"] == "cpu" else "cpu")

    train_dir = config["deepglobe"]["train_dir"]
    image_size = config["model"]["input_size"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    lr = config["model"]["learning_rate"]
    val_split = config["training"]["val_split"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Device: {device}")

    full_dataset = RoadDataset(train_dir, image_size=image_size, augment=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    model = UNetResNet34(pretrained=config["model"]["pretrained"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")


if __name__ == "__main__":
    train()