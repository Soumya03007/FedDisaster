#!/usr/bin/env python3
"""Run single-image inference with the verified EfficientNet + PCA + RF artifacts."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import torch
from PIL import Image

from dataset_loader import build_transforms
from models import EfficientNetB0Extractor
from utils import get_device


DEFAULT_CLASS_NAMES = [
    "Damaged_Infrastructure",
    "Fire_Disaster",
    "Human_Damage",
    "Land_Disaster",
    "Non_Damage",
    "Water_Disaster",
]

RELEASE_URL = "https://github.com/Soumya03007/FedDisaster/releases/tag/v0.1.0-efficientnet-rf"


def _resolve_artifacts(artifacts: str) -> Dict[str, Path]:
    if artifacts == "release":
        root = Path(".")
    else:
        root = Path(artifacts)

    paths = {
        "backbone": root / "global_cnn.pt",
        "pca": root / "global_pca.pkl",
        "rf": root / "global_rf.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        message = [
            "Missing artifact files:",
            *[f"  - {path}" for path in missing],
            "",
            "Download the release artifact bundle, extract it, and rerun this command.",
            f"Release: {RELEASE_URL}",
        ]
        raise FileNotFoundError("\n".join(message))
    return paths


def _class_names_from_root(class_root: str) -> List[str]:
    root = Path(class_root)
    if root.is_dir():
        names = sorted(path.name for path in root.iterdir() if path.is_dir())
        if names:
            return names
    return DEFAULT_CLASS_NAMES


def _load_image(image_path: Path) -> torch.Tensor:
    _, transform = build_transforms(preset="efficientnet_b0")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return transform(image).unsqueeze(0)


def predict(args: argparse.Namespace) -> Dict:
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    artifact_paths = _resolve_artifacts(args.artifacts)
    class_names = _class_names_from_root(args.class_root)
    device = get_device()

    backbone = EfficientNetB0Extractor(pretrained=False).to(device)
    backbone.load_state_dict(torch.load(artifact_paths["backbone"], map_location=device))
    backbone.eval()

    pca = joblib.load(artifact_paths["pca"])
    rf = joblib.load(artifact_paths["rf"])

    image_tensor = _load_image(image_path).to(device)
    with torch.no_grad():
        features = backbone(image_tensor).cpu().numpy()

    features_pca = pca.transform(features)
    predicted_index = int(rf.predict(features_pca)[0])

    probabilities = None
    if hasattr(rf, "predict_proba"):
        probabilities = rf.predict_proba(features_pca)[0]

    label = class_names[predicted_index] if predicted_index < len(class_names) else str(predicted_index)
    result = {
        "image": str(image_path),
        "prediction": label,
        "class_index": predicted_index,
        "artifacts": {
            "backbone": str(artifact_paths["backbone"]),
            "pca": str(artifact_paths["pca"]),
            "rf": str(artifact_paths["rf"]),
        },
    }

    if probabilities is not None:
        top_k = min(int(args.top_k), len(probabilities))
        ranked = sorted(enumerate(probabilities), key=lambda item: float(item[1]), reverse=True)[:top_k]
        result["top_k"] = [
            {
                "class_index": int(index),
                "label": class_names[index] if index < len(class_names) else str(index),
                "probability": float(probability),
            }
            for index, probability in ranked
        ]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to one image to classify.")
    parser.add_argument(
        "--artifacts",
        default="release",
        help="Artifact directory, or 'release' to use global_cnn.pt/global_pca.pkl/global_rf.pkl in the repo root.",
    )
    parser.add_argument(
        "--class_root",
        default="data/global_test",
        help="Folder whose class subdirectories define class index order.",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of ranked predictions to show.")
    args = parser.parse_args()

    print(json.dumps(predict(args), indent=2))


if __name__ == "__main__":
    main()
