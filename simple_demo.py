#!/usr/bin/env python3
"""
Federated Learning Demo with Global Random Forest Classifier
CNN is trained via FedAvg
Random Forest is trained centrally on extracted CNN features (REALISTIC)
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import time
import joblib
from datetime import datetime

from models import SimpleCNN, EfficientNetB0Extractor, LocalHead
from dataset_loader import load_imagefolder_dataloaders, load_global_test_loader
from utils import get_device, get_parameters_from_model, set_parameters_to_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# ✅ GLOBAL RF TRAIN + EVAL (NO DATA LEAKAGE)
# ---------------------------------------------------------
def train_and_evaluate_rf(cnn_model, client_loaders, global_test_loader, device):
    cnn_model.eval()
    X_train, y_train = [], []

    # Collect features from all client train sets
    with torch.no_grad():
        for train_loader, _ in client_loaders:
            for images, labels in train_loader:
                images = images.to(device)
                features = cnn_model(images)
                X_train.append(features.cpu().numpy())
                y_train.append(labels.numpy())

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_train += 0.01 * np.random.randn(*X_train.shape)  # Feature noise

    # PCA (retain 90% variance)
    pca = PCA(n_components=0.90, whiten=True)
    X_train_pca = pca.fit_transform(X_train)

    # Random Forest (regularized)
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train_pca, y_train)

    # Evaluate on global test set
    X_test, y_test = [], []
    with torch.no_grad():
        for images, labels in global_test_loader:
            images = images.to(device)
            features = cnn_model(images)
            X_test.append(features.cpu().numpy())
            y_test.append(labels.numpy())

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    X_test_pca = pca.transform(X_test)

    acc = rf.score(X_test_pca, y_test)
    print(f"[GLOBAL RF ACCURACY: {acc:.4f}]")

    # Save models
    joblib.dump(rf, "global_rf.pkl")
    joblib.dump(pca, "global_pca.pkl")

    return acc


# ---------------------------------------------------------
# ✅ FEDERATED ROUND (CNN ONLY)
# ---------------------------------------------------------
def simulate_federated_round(client_models, client_loaders, global_model, device, criterion, round_num, local_heads, optimizers):
    print(f"\n[FEDERATED ROUND {round_num}]")
    print("=" * 60)

    # Get current global parameters
    global_params = get_parameters_from_model(global_model)

    client_updates = []
    client_sizes = []

    # ---- CLIENT SIDE TRAINING ----
    for cid, (model, head, optim, (train_loader, _)) in enumerate(
        zip(client_models, local_heads, optimizers, client_loaders), 1
    ):
        print(f"\n[CLIENT {cid} LOCAL TRAINING]")

        # Reset CNN weights to global
        set_parameters_to_model(model, global_params)
        model.train()
        head.train()

        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()

            # Forward: CNN → Local head
            features = model(images)
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        print(f"   Local train loss: {train_loss:.4f}")

        client_updates.append(get_parameters_from_model(model))
        client_sizes.append(len(train_loader.dataset))

    # ---- FEDAVG AGGREGATION ----
    total_samples = sum(client_sizes)
    aggregated_params = []
    # Aggregate in float64 to avoid dtype casting errors, then cast back
    for i in range(len(client_updates[0])):
        orig_dtype = client_updates[0][i].dtype
        weighted_sum = np.zeros_like(client_updates[0][i], dtype=np.float64)
        for update, size in zip(client_updates, client_sizes):
            weighted_sum += (size / total_samples) * update[i].astype(np.float64)
        # Cast back to original dtype (e.g., float32 or int64) to match model state_dict
        aggregated_params.append(weighted_sum.astype(orig_dtype))

    set_parameters_to_model(global_model, aggregated_params)
    print(f"[SERVER AGGREGATION COMPLETE]")


# ---------------------------------------------------------
# ✅ MAIN
# ---------------------------------------------------------
def _set_backbone_trainable(backbone_model: nn.Module, train_backbone: bool):
    """CPU-friendly default: freeze backbone unless explicitly requested."""
    for p in backbone_model.parameters():
        p.requires_grad = bool(train_backbone)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        default="simplecnn",
        choices=["simplecnn", "efficientnet_b0"],
        help="Feature extractor backbone",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of clients (expects data/client_1..data/client_N)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for client loaders and global test loader",
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="Train the backbone on CPU (SLOW for EfficientNet). Default is frozen backbone.",
    )
    args = parser.parse_args()

    # Train backbone by default for SimpleCNN; freeze by default for EfficientNet (CPU practicality)
    train_backbone = True if args.backbone == "simplecnn" else bool(args.train_backbone)

    print("🌊 FEDERATED LEARNING + RANDOM FOREST (LEARNING ACROSS ROUNDS)")
    print("=" * 70)
    print(f"Backbone: {args.backbone} | num_clients: {args.num_clients} | batch_size: {args.batch_size} | train_backbone: {train_backbone}")

    device = get_device()
    criterion = nn.CrossEntropyLoss()

    preset = "efficientnet_b0" if args.backbone == "efficientnet_b0" else "simplecnn"

    # ---- LOAD CLIENT DATA ----
    print("📁 LOADING CLIENT DATA:")
    client_loaders = []

    num_classes = None
    class_to_idx_ref = None

    for cid in range(1, args.num_clients + 1):
        train_loader, test_loader, client_num_classes = load_imagefolder_dataloaders(
            f"data/client_{cid}/train",
            f"data/client_{cid}/test",
            batch_size=args.batch_size,
            preset=preset,
        )

        # Ensure consistent label mapping across clients (CRITICAL for multi-client training)
        class_to_idx = train_loader.dataset.class_to_idx
        if num_classes is None:
            num_classes = client_num_classes
            class_to_idx_ref = class_to_idx
            print(f"   Detected classes ({num_classes}): {list(class_to_idx_ref.keys())}")
        else:
            if client_num_classes != num_classes:
                raise ValueError(
                    f"Client {cid} num_classes={client_num_classes} differs from expected {num_classes}. "
                    "Ensure every client has the same class folders under train/ and test/."
                )
            if class_to_idx != class_to_idx_ref:
                raise ValueError(
                    f"Client {cid} class_to_idx differs from other clients. "
                    "This usually happens when class folders are missing or named differently. "
                    "Ensure every client has the exact same class folder names (empty folders are OK)."
                )

        client_loaders.append((train_loader, test_loader))
        print(f"   Client {cid}: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # ---- LOAD GLOBAL TEST ----
    global_test_loader, global_num_classes = load_global_test_loader(
        "data/global_test",
        batch_size=args.batch_size,
        preset=preset,
    )
    if global_num_classes != num_classes:
        raise ValueError(
            f"Global test num_classes={global_num_classes} differs from clients num_classes={num_classes}. "
            "Ensure data/global_test has the same class folders as the clients."
        )

    if hasattr(global_test_loader.dataset, "class_to_idx") and global_test_loader.dataset.class_to_idx != class_to_idx_ref:
        raise ValueError(
            "Global test class_to_idx differs from clients. "
            "Ensure data/global_test uses the exact same class folder names as the clients."
        )

    print(f"   Global test: {len(global_test_loader.dataset)} images")

    # ---- INITIALIZE MODELS ----
    if args.backbone == "efficientnet_b0":
        global_model = EfficientNetB0Extractor(pretrained=True).to(device)
        client_models = [EfficientNetB0Extractor(pretrained=True).to(device) for _ in client_loaders]
    else:
        global_model = SimpleCNN().to(device)
        client_models = [SimpleCNN().to(device) for _ in client_loaders]

    # Freeze backbone by default on CPU (EfficientNet is heavy)
    _set_backbone_trainable(global_model, train_backbone)
    for m in client_models:
        _set_backbone_trainable(m, train_backbone)

    # Local heads must output the GLOBAL number of classes, even if each client only sees a subset
    local_heads = [LocalHead(global_model.feature_dim, num_classes).to(device) for _ in client_models]

    # Optimizer: always train local head; optionally train backbone
    optimizers = []
    for model, head in zip(client_models, local_heads):
        params = list(head.parameters())
        if train_backbone:
            params += [p for p in model.parameters() if p.requires_grad]
        optimizers.append(torch.optim.Adam(params, lr=1e-3 if args.backbone == "simplecnn" else 1e-4))

    print(f"\n[Initialized CNN feature extractor and client heads]")

    # ---- METRICS FOR STREAMLIT ----
    def update_streamlit_metrics(accuracies, training_complete=False):
        metrics = {
            "accuracies": accuracies,
            "training_complete": training_complete,
            "last_updated": datetime.now().isoformat(),
            "rounds_expected": 5,
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # ---- FEDERATED TRAINING ROUNDS ----
    num_rounds = 5
    accuracies = []

    for round_num in range(1, num_rounds + 1):
        simulate_federated_round(
            client_models, client_loaders, global_model, device, criterion,
            round_num, local_heads, optimizers
        )

        # Evaluate global RF after each round
        global_acc = train_and_evaluate_rf(global_model, client_loaders, global_test_loader, device)
        accuracies.append(global_acc)

        print(f"\n[GLOBAL RF ACCURACY AFTER ROUND {round_num}: {global_acc:.4f}]")
        update_streamlit_metrics(accuracies, training_complete=(round_num == num_rounds))
        time.sleep(3)

    torch.save(global_model.state_dict(), "global_cnn.pt")
    print(f"\n[FINAL GLOBAL RF ACCURACY: {accuracies[-1]:.4f}]")
    print(f"[OK] Global CNN saved: global_cnn.pt")
    print(f"[OK] Global RF saved: global_rf.pkl")
    print(f"[OK] Global PCA saved: global_pca.pkl")
    print(f"[OK] Streamlit ready")


if __name__ == "__main__":
    main()
