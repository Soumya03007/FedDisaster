#!/usr/bin/env python3
"""
Local federated learning demo with a centralized Random Forest classifier.

- Simulates FedAvg in-process without Flower networking
- Defaults to EfficientNet-B0 to mirror the preferred modern path
- Keeps SimpleCNN available as a compatibility option
"""

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from dataset_loader import load_global_test_loader, load_imagefolder_dataloaders
from models import EfficientNetB0Extractor, LocalHead, SimpleCNN
from utils import get_device, get_parameters_from_model, set_parameters_to_model


def train_and_evaluate_rf(cnn_model, client_loaders, global_test_loader, device):
    """Train a centralized PCA+RF classifier on extracted backbone features."""
    cnn_model.eval()
    x_train, y_train = [], []

    with torch.no_grad():
        for train_loader, _ in client_loaders:
            for images, labels in train_loader:
                images = images.to(device)
                features = cnn_model(images)
                x_train.append(features.cpu().numpy())
                y_train.append(labels.numpy())

    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    pca = PCA(n_components=0.90, whiten=True)
    x_train_pca = pca.fit_transform(x_train)

    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(x_train_pca, y_train)

    x_test, y_test = [], []
    with torch.no_grad():
        for images, labels in global_test_loader:
            images = images.to(device)
            features = cnn_model(images)
            x_test.append(features.cpu().numpy())
            y_test.append(labels.numpy())

    x_test = np.vstack(x_test)
    y_test = np.hstack(y_test)
    x_test_pca = pca.transform(x_test)

    acc = float(rf.score(x_test_pca, y_test))
    print(f"[GLOBAL RF ACCURACY: {acc:.4f}]")

    joblib.dump(rf, "global_rf.pkl")
    joblib.dump(pca, "global_pca.pkl")
    return acc


def simulate_federated_round(
    client_models,
    client_loaders,
    global_model,
    device,
    criterion,
    round_num,
    local_heads,
    optimizers,
    epochs,
    train_backbone,
):
    """Run one in-process federated round and aggregate model updates with FedAvg."""
    print(f"\n[FEDERATED ROUND {round_num}]")
    print("=" * 60)

    global_params = get_parameters_from_model(global_model)
    client_updates = []
    client_sizes = []

    for cid, (model, head, optimizer, (train_loader, _)) in enumerate(
        zip(client_models, local_heads, optimizers, client_loaders), start=1
    ):
        print(f"\n[CLIENT {cid} LOCAL TRAINING]")
        set_parameters_to_model(model, global_params)
        model.train()
        head.train()

        for epoch in range(epochs):
            running_loss = 0.0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if train_backbone:
                    features = model(images)
                else:
                    with torch.no_grad():
                        features = model(images)

                outputs = head(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                total += images.size(0)

            avg_loss = running_loss / (total + 1e-12)
            print(f"   Epoch {epoch + 1}/{epochs} train loss: {avg_loss:.4f}")

        client_updates.append(get_parameters_from_model(model))
        client_sizes.append(len(train_loader.dataset))

    total_samples = sum(client_sizes)
    aggregated_params = []
    for index in range(len(client_updates[0])):
        original_dtype = client_updates[0][index].dtype
        weighted_sum = np.zeros_like(client_updates[0][index], dtype=np.float64)
        for update, size in zip(client_updates, client_sizes):
            weighted_sum += (size / total_samples) * update[index].astype(np.float64)
        aggregated_params.append(weighted_sum.astype(original_dtype))

    set_parameters_to_model(global_model, aggregated_params)
    print("[SERVER AGGREGATION COMPLETE]")


def _set_backbone_trainable(backbone_model: nn.Module, train_backbone: bool):
    for parameter in backbone_model.parameters():
        parameter.requires_grad = bool(train_backbone)


def _build_backbone(backbone: str) -> nn.Module:
    if backbone == "efficientnet_b0":
        return EfficientNetB0Extractor(pretrained=True)
    return SimpleCNN()


def _preset_for_backbone(backbone: str) -> str:
    if backbone == "efficientnet_b0":
        return "efficientnet_b0"
    return "simplecnn"


def update_streamlit_metrics(accuracies, training_complete=False, round_num=None, args=None):
    status = "completed" if training_complete else "started"
    metrics = {
        "accuracies": accuracies,
        "status": status,
        "round_num": round_num or len(accuracies),
        "training_complete": training_complete,
        "last_updated": datetime.now().isoformat(),
        "rounds_expected": args.num_rounds if args else 5,
        "backbone": args.backbone if args else "efficientnet_b0",
    }
    with open("metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["simplecnn", "efficientnet_b0"],
        help="Feature extractor backbone (default: efficientnet_b0; simplecnn kept for compatibility)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of clients (expects data/client_1..data/client_N)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of simulated federated rounds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Local epochs per client per round",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for client loaders and global test loader",
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="Fine-tune the shared backbone. EfficientNet-B0 stays frozen by default for CPU practicality.",
    )
    parser.add_argument(
        "--cpu_safe",
        action="store_true",
        default=True,
        help="Use CPU-safe DataLoader settings (num_workers=0, pin_memory=False). Recommended for Windows CPU.",
    )
    args = parser.parse_args()

    train_backbone = True if args.backbone == "simplecnn" else bool(args.train_backbone)

    print("FEDERATED LEARNING + RANDOM FOREST DEMO")
    print("=" * 70)
    print(
        f"Backbone: {args.backbone} | num_clients: {args.num_clients} | "
        f"num_rounds: {args.num_rounds} | epochs: {args.epochs} | "
        f"batch_size: {args.batch_size} | train_backbone: {train_backbone}"
    )

    device = get_device()
    criterion = nn.CrossEntropyLoss()
    preset = _preset_for_backbone(args.backbone)

    print("LOADING CLIENT DATA:")
    client_loaders = []
    num_classes = None
    class_to_idx_ref = None

    for cid in range(1, args.num_clients + 1):
        train_loader, test_loader, client_num_classes = load_imagefolder_dataloaders(
            f"data/client_{cid}/train",
            f"data/client_{cid}/test",
            batch_size=args.batch_size,
            preset=preset,
            optimized=not args.cpu_safe,
        )

        class_to_idx = train_loader.dataset.class_to_idx
        if num_classes is None:
            num_classes = client_num_classes
            class_to_idx_ref = class_to_idx
            print(f"   Detected classes ({num_classes}): {list(class_to_idx_ref.keys())}")
        else:
            if client_num_classes != num_classes:
                raise ValueError(
                    f"Client {cid} num_classes={client_num_classes} differs from expected {num_classes}."
                )
            if class_to_idx != class_to_idx_ref:
                raise ValueError(
                    f"Client {cid} class_to_idx differs from the other clients. "
                    "Ensure class folder names are identical across all client folders."
                )

        client_loaders.append((train_loader, test_loader))
        print(f"   Client {cid}: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    global_test_loader, global_num_classes = load_global_test_loader(
        "data/global_test",
        batch_size=args.batch_size,
        preset=preset,
        optimized=not args.cpu_safe,
    )
    if global_num_classes != num_classes:
        raise ValueError(
            f"Global test num_classes={global_num_classes} differs from clients num_classes={num_classes}."
        )
    if (
        hasattr(global_test_loader.dataset, "class_to_idx")
        and global_test_loader.dataset.class_to_idx != class_to_idx_ref
    ):
        raise ValueError(
            "Global test class_to_idx differs from the client datasets. "
            "Ensure class folder names are identical."
        )

    print(f"   Global test: {len(global_test_loader.dataset)} images")

    global_model = _build_backbone(args.backbone).to(device)
    client_models = [_build_backbone(args.backbone).to(device) for _ in client_loaders]

    _set_backbone_trainable(global_model, train_backbone)
    for model in client_models:
        _set_backbone_trainable(model, train_backbone)

    local_heads = [LocalHead(global_model.feature_dim, num_classes).to(device) for _ in client_models]

    optimizers = []
    for model, head in zip(client_models, local_heads):
        params = list(head.parameters())
        if train_backbone:
            params += [parameter for parameter in model.parameters() if parameter.requires_grad]
        lr = 1e-4 if args.backbone == "efficientnet_b0" else 1e-3
        optimizers.append(torch.optim.Adam(params, lr=lr))

    # Initial status update BEFORE any training
    print("[DASHBOARD] Sending started signal...")
    update_streamlit_metrics([], False, 0, args)
    print("[DASHBOARD] Status sent - Streamlit should show 'Started' now!")

    accuracies = []
    for round_num in range(1, args.num_rounds + 1):
        simulate_federated_round(
            client_models=client_models,
            client_loaders=client_loaders,
            global_model=global_model,
            device=device,
            criterion=criterion,
            round_num=round_num,
            local_heads=local_heads,
            optimizers=optimizers,
            epochs=args.epochs,
            train_backbone=train_backbone,
        )

        global_acc = train_and_evaluate_rf(global_model, client_loaders, global_test_loader, device)
        accuracies.append(global_acc)

        print(f"\n[GLOBAL RF ACCURACY AFTER ROUND {round_num}: {global_acc:.4f}]")
        update_streamlit_metrics(accuracies, training_complete=(round_num == args.num_rounds), round_num=round_num, args=args)
        time.sleep(3)

    torch.save(global_model.state_dict(), "global_cnn.pt")
    print(f"\n[FINAL GLOBAL RF ACCURACY: {accuracies[-1]:.4f}]")
    print("[OK] Global CNN saved: global_cnn.pt")
    print("[OK] Global RF saved: global_rf.pkl")
    print("[OK] Global PCA saved: global_pca.pkl")
    print("[OK] Streamlit ready")


if __name__ == "__main__":
    main()

