#!/usr/bin/env python3
"""Evaluate saved EfficientNet backbone + PCA/RF artifacts on global_test."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_loader import load_global_test_loader
from models import EfficientNetB0Extractor
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", default="best_artifacts/global_backbone_best.pt")
    parser.add_argument("--rf_path", default="best_artifacts/global_rf_best.pkl")
    parser.add_argument("--pca_path", default="best_artifacts/global_pca_best.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    model = EfficientNetB0Extractor(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.backbone_path, map_location=device))
    model.eval()

    pca = joblib.load(args.pca_path)
    rf = joblib.load(args.rf_path)
    global_test_loader, num_classes = load_global_test_loader(
        "data/global_test",
        batch_size=args.batch_size,
        preset="efficientnet_b0",
        optimized=False,
    )

    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in global_test_loader:
            images = images.to(device)
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())

    x_test = np.vstack(features)
    y_test = np.hstack(labels)
    x_test_pca = pca.transform(x_test)
    accuracy = float(rf.score(x_test_pca, y_test))

    print(f"classes={num_classes}")
    print(f"samples={len(y_test)}")
    print(f"accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
