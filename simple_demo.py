#!/usr/bin/env python3
"""
Federated Learning Demo with Global Random Forest Classifier
CNN is trained via FedAvg
Random Forest is trained centrally on extracted CNN features (REALISTIC)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import joblib
from datetime import datetime

from models import SimpleCNN, LocalHead
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
    print(f"🎯 GLOBAL RF ACCURACY: {acc:.4f}")

    # Save models
    joblib.dump(rf, "global_rf.pkl")
    joblib.dump(pca, "global_pca.pkl")

    return acc


# ---------------------------------------------------------
# ✅ FEDERATED ROUND (CNN ONLY)
# ---------------------------------------------------------
def simulate_federated_round(client_models, client_loaders, global_model, device, criterion, round_num, local_heads, optimizers):
    print(f"\n🔄 FEDERATED ROUND {round_num}")
    print("=" * 60)

    # Get current global parameters
    global_params = get_parameters_from_model(global_model)

    client_updates = []
    client_sizes = []

    # ---- CLIENT SIDE TRAINING ----
    for cid, (model, head, optim, (train_loader, _)) in enumerate(
        zip(client_models, local_heads, optimizers, client_loaders), 1
    ):
        print(f"\n📱 CLIENT {cid} LOCAL TRAINING:")

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
    for i in range(len(client_updates[0])):
        weighted_sum = np.zeros_like(client_updates[0][i])
        for update, size in zip(client_updates, client_sizes):
            weighted_sum += (size / total_samples) * update[i]
        aggregated_params.append(weighted_sum)

    set_parameters_to_model(global_model, aggregated_params)
    print(f"🌐 SERVER AGGREGATION COMPLETE")


# ---------------------------------------------------------
# ✅ MAIN
# ---------------------------------------------------------
def main():
    print("🌊 FEDERATED LEARNING + RANDOM FOREST (LEARNING ACROSS ROUNDS)")
    print("=" * 70)

    device = get_device()
    criterion = nn.CrossEntropyLoss()

    # ---- LOAD CLIENT DATA ----
    print("📁 LOADING CLIENT DATA:")
    client_loaders = []
    for cid in [1, 2, 3]:
        train_loader, test_loader, _ = load_imagefolder_dataloaders(
            f"data/client_{cid}/train",
            f"data/client_{cid}/test",
            batch_size=8,
        )
        client_loaders.append((train_loader, test_loader))
        print(f"   Client {cid}: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # ---- LOAD GLOBAL TEST ----
    global_test_loader, _ = load_global_test_loader("data/global_test", batch_size=8)
    print(f"   Global test: {len(global_test_loader.dataset)} images")

    # ---- INITIALIZE MODELS ----
    global_model = SimpleCNN().to(device)
    client_models = [SimpleCNN().to(device) for _ in client_loaders]
    local_heads = [LocalHead(global_model.feature_dim, 2).to(device) for _ in client_models]
    optimizers = [torch.optim.Adam(list(model.parameters()), lr=1e-3) for model in client_models]

    print(f"\n🤖 Initialized CNN feature extractor and client heads")

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

        print(f"\n🎯 GLOBAL RF ACCURACY AFTER ROUND {round_num}: {global_acc:.4f}")
        update_streamlit_metrics(accuracies, training_complete=(round_num == num_rounds))
        time.sleep(3)

    torch.save(global_model.state_dict(), "global_cnn.pt")
    print(f"\n🏆 FINAL GLOBAL RF ACCURACY: {accuracies[-1]:.4f}")
    print(f"✅ Global CNN saved: global_cnn.pt")
    print(f"✅ Global RF saved: global_rf.pkl")
    print(f"✅ Global PCA saved: global_pca.pkl")
    print(f"✅ Streamlit ready")


if __name__ == "__main__":
    main()
