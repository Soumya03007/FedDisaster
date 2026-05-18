import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, Subset

from typing import Optional

from dataset_loader import load_imagefolder_dataloaders
from models import EfficientNetB0Extractor, SimpleCNN, LocalHead
from utils import (
    configure_backbone_training,
    get_device,
    get_parameters_from_model,
    payload_size_kb,
    set_global_seeds,
    get_trainable_state_keys,
    set_parameters_to_model,
    sparse_payload_size_kb,
    sparsify_parameters,
)

try:
    from opacus import PrivacyEngine
except Exception:
    PrivacyEngine = None


def _preset_for_backbone(backbone: str) -> str:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return "efficientnet_b0"
    return "simplecnn"


def _build_backbone(backbone: str) -> torch.nn.Module:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return EfficientNetB0Extractor(pretrained=True)
    return SimpleCNN()


def get_loaders_for_client(cid: int, batch_size: int, preset: str):
    train_dir = f"data/client_{cid}/train"
    test_dir = f"data/client_{cid}/test"
    return load_imagefolder_dataloaders(train_dir, test_dir, batch_size=batch_size, preset=preset)


class FlowerClient(fl.client.NumPyClient):
    """Federated client.

    - Receives global backbone feature extractor
    - Trains a LOCAL classification head
    - Optionally fine-tunes the SHARED backbone (real FedAvg)
    - Sends back ONLY the backbone parameters (FedAvg)
    """

    def __init__(
        self,
        cid: int,
        batch_size: int = 32,
        lr: float = 1e-3,
        backbone: str = "simplecnn",
        train_backbone: bool = True,
        backbone_lr: Optional[float] = None,
        trainable_blocks: int = 1,
        max_trainable_blocks: int = 1,
    ):
        set_global_seeds(42)
        self.cid = cid
        self.batch_size = batch_size
        self.lr = lr
        self.backbone = backbone
        self.train_backbone = bool(train_backbone)
        self.trainable_blocks = int(trainable_blocks)
        self.max_trainable_blocks = max(int(max_trainable_blocks), int(trainable_blocks))
        self.preset = _preset_for_backbone(backbone)

        # ---- Data ----
        self.train_loader, self.test_loader, self.num_classes = get_loaders_for_client(cid, batch_size, preset=self.preset)
        self._round_sample_order = np.array([], dtype=np.int64)
        self._round_sample_cursor = 0
        self._round_pass_index = 0

        # ---- Device ----
        self.device = get_device()

        # ---- Global Federated Model (Feature Extractor) ----
        self.model = _build_backbone(backbone).to(self.device)

        # ---- Local Head (NOT SHARED) ----
        self.local_head = LocalHead(self.model.feature_dim, self.num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.privacy_engine = None
        self.dp_enabled = False

        # ---- Optimizer ----
        # Always train local head; optionally train backbone.
        head_params = list(self.local_head.parameters())
        self.head_optimizer = optim.Adam(head_params, lr=self.lr)
        self.backbone_optimizer = None
        configure_backbone_training(
            self.model,
            backbone=self.backbone,
            train_backbone=self.train_backbone,
            trainable_blocks=self.max_trainable_blocks,
        )
        self.shared_state_keys = get_trainable_state_keys(self.model)
        self.backbone_lr = float(1e-4 if backbone_lr is None and self.preset == "efficientnet_b0" else (backbone_lr if backbone_lr is not None else self.lr))
        trainable_backbone_params = self._refresh_trainable_blocks(self.trainable_blocks)

        print(
            f"[Client {self.cid}] device={self.device} | backbone={self.backbone} | "
            f"train_backbone={self.train_backbone} | trainable_blocks={self.trainable_blocks}/{self.max_trainable_blocks} | "
            f"trainable_backbone_params={trainable_backbone_params} | "
            f"shared_tensors={len(self.shared_state_keys)}"
        )

        self.optimizer = self.head_optimizer

    def _refresh_trainable_blocks(self, trainable_blocks: int) -> int:
        self.trainable_blocks = max(1, min(int(trainable_blocks), int(self.max_trainable_blocks)))
        trainable_backbone_params = configure_backbone_training(
            self.model,
            backbone=self.backbone,
            train_backbone=self.train_backbone,
            trainable_blocks=self.trainable_blocks,
        )
        self.active_parameter_names = [
            name for name, parameter in self._base_model().named_parameters() if parameter.requires_grad
        ]
        if trainable_backbone_params > 0:
            self.backbone_optimizer = optim.Adam(
                [p for p in self._base_model().parameters() if p.requires_grad],
                lr=float(self.backbone_lr),
            )
        else:
            self.backbone_optimizer = None
            self.train_backbone = False
        return trainable_backbone_params

    def _reset_round_sample_order(self) -> None:
        dataset_size = len(self.train_loader.dataset)
        self._round_sample_order = np.random.permutation(dataset_size)
        self._round_sample_cursor = 0
        self._round_pass_index += 1

    def _build_round_train_loader(self, max_batches_per_round: int):
        if max_batches_per_round <= 0:
            return self.train_loader, {
                "mode": "full",
                "pass_index": self._round_pass_index,
                "start_sample": 1,
                "end_sample": len(self.train_loader.dataset),
                "dataset_size": len(self.train_loader.dataset),
                "completed_pass": False,
            }

        dataset = self.train_loader.dataset
        if len(dataset) == 0:
            return self.train_loader, {
                "mode": "empty",
                "pass_index": self._round_pass_index,
                "start_sample": 0,
                "end_sample": 0,
                "dataset_size": 0,
                "completed_pass": False,
            }

        if len(self._round_sample_order) != len(dataset) or self._round_sample_cursor >= len(dataset):
            self._reset_round_sample_order()

        sample_limit = max(1, int(max_batches_per_round) * int(self.batch_size))
        start_idx = self._round_sample_cursor
        end_idx = min(self._round_sample_cursor + sample_limit, len(dataset))
        selected_indices = self._round_sample_order[self._round_sample_cursor:end_idx].tolist()
        self._round_sample_cursor = end_idx
        completed_pass = self._round_sample_cursor >= len(dataset)

        subset = Subset(dataset, selected_indices)
        return (
            DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.train_loader.num_workers,
                pin_memory=self.train_loader.pin_memory,
            ),
            {
                "mode": "subset",
                "pass_index": self._round_pass_index,
                "start_sample": start_idx + 1,
                "end_sample": end_idx,
                "dataset_size": len(dataset),
                "completed_pass": completed_pass,
            },
        )

    def _base_model(self) -> torch.nn.Module:
        return self.model._module if hasattr(self.model, "_module") else self.model

    def _enable_private_backbone(
        self,
        epochs: int,
        target_epsilon: float,
        target_delta: float,
        max_grad_norm: float,
    ):
        if PrivacyEngine is None:
            raise ImportError("Opacus is required for differential privacy. Install it with: pip install opacus")
        if self.backbone_optimizer is None:
            raise ValueError("Differential privacy requires train_backbone=True so the shared model is trainable.")
        if self.dp_enabled:
            return self.model, self.backbone_optimizer, self.train_loader

        privacy_engine = PrivacyEngine()
        self.model, self.backbone_optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
            module=self._base_model(),
            optimizer=self.backbone_optimizer,
            data_loader=self.train_loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )
        self.privacy_engine = privacy_engine
        self.dp_enabled = True
        return self.model, self.backbone_optimizer, self.train_loader

    # Flower will call this to get current local weights (CNN ONLY)
    def get_parameters(self, config):
        return get_parameters_from_model(self._base_model(), state_keys=self.shared_state_keys)

    def fit(self, parameters, config):
        set_global_seeds(42)
        round_started_at = time.perf_counter()
        # ---- Load global CNN parameters ----
        if parameters is not None and len(parameters) > 0:
            set_parameters_to_model(self._base_model(), parameters, state_keys=self.shared_state_keys)

        self.model.train()
        self.local_head.train()

        # Read training config from server
        epochs = int(config.get("epochs", 1))
        batch_size = int(config.get("batch_size", self.batch_size))
        use_sparsification = bool(config.get("use_sparsification", False))
        sparsify_k = float(config.get("sparsify_k", 0.01))
        use_dp = bool(config.get("use_dp", False))
        target_epsilon = float(config.get("target_epsilon", 8.0))
        target_delta = float(config.get("target_delta", 1e-5))
        max_grad_norm = float(config.get("max_grad_norm", 1.2))
        use_fedprox = bool(config.get("use_fedprox", False))
        proximal_mu = float(config.get("fedprox_mu", 0.01))
        max_batches_per_round = int(config.get("max_batches_per_round", 0))
        round_trainable_blocks = int(config.get("trainable_blocks", self.trainable_blocks))

        if batch_size != self.batch_size:
            self.train_loader, self.test_loader, _ = get_loaders_for_client(self.cid, batch_size, preset=self.preset)
            self.batch_size = batch_size
            self._round_sample_order = np.array([], dtype=np.int64)
            self._round_sample_cursor = 0
            self._round_pass_index = 0
            self.dp_enabled = False
            self.privacy_engine = None
            self._refresh_trainable_blocks(self.trainable_blocks)

        if round_trainable_blocks != self.trainable_blocks:
            trainable_backbone_params = self._refresh_trainable_blocks(round_trainable_blocks)
            print(
                f"[Client {self.cid}] Progressive unfreeze update -> trainable_blocks={self.trainable_blocks}/"
                f"{self.max_trainable_blocks} | trainable_backbone_params={trainable_backbone_params}"
            )

        if use_dp:
            self._enable_private_backbone(
                epochs=epochs,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
            )

        global_params_tensor = None
        if use_fedprox and parameters is not None and len(parameters) > 0:
            incoming_param_map = dict(zip(self.shared_state_keys, parameters))
            global_params_tensor = [
                torch.tensor(incoming_param_map[name], device=self.device, dtype=local_param.dtype)
                for name, local_param in self._base_model().named_parameters()
                if name in self.active_parameter_names
            ]

        # ---- Local Training (CNN frozen, Head trained) ----
        round_train_loader, round_slice_info = self._build_round_train_loader(max_batches_per_round=max_batches_per_round)
        if round_slice_info["mode"] == "subset":
            print(
                f"[Client {self.cid}] Round data pass={round_slice_info['pass_index']} | "
                f"samples={round_slice_info['start_sample']}-{round_slice_info['end_sample']} "
                f"of {round_slice_info['dataset_size']}"
            )
        for epoch in range(epochs):
            running_loss = 0.0
            total = 0
            batches_processed = 0

            for images, labels in round_train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.head_optimizer.zero_grad()
                if self.backbone_optimizer is not None:
                    self.backbone_optimizer.zero_grad()

                # Backbone feature extraction (optionally trainable)
                if self.train_backbone:
                    features = self.model(images)
                else:
                    with torch.no_grad():
                        features = self.model(images)

                # Classification via local head
                logits = self.local_head(features)
                ce_loss = self.criterion(logits, labels)
                loss = ce_loss
                if use_fedprox and global_params_tensor is not None and self.train_backbone:
                    proximal_term = sum(
                        torch.norm(weight - global_weight) ** 2
                        for weight, global_weight in zip(self._base_model().parameters(), global_params_tensor)
                    )
                    loss = ce_loss + (proximal_mu / 2.0) * proximal_term

                loss.backward()
                self.head_optimizer.step()
                if self.backbone_optimizer is not None:
                    self.backbone_optimizer.step()

                running_loss += loss.item() * images.size(0)
                total += images.size(0)
                batches_processed += 1

            avg_loss = running_loss / (total + 1e-12)
            batch_note = (
                f", batches={batches_processed}, samples={total}"
                if max_batches_per_round > 0
                else f", batches={batches_processed}"
            )
            print(f"[Client {self.cid}] Epoch {epoch+1}/{epochs} train loss: {avg_loss:.4f}{batch_note}")

        if round_slice_info["mode"] == "subset" and round_slice_info["completed_pass"]:
            print(
                f"[Client {self.cid}] Completed local data pass {round_slice_info['pass_index']} "
                f"over {round_slice_info['dataset_size']} samples; next round will reshuffle."
            )

        # ---- Local Evaluation (for monitoring only) ----
        self.model.eval()
        self.local_head.eval()

        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.model(images)
                logits = self.local_head(features)

                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=1)

                test_loss += loss.item() * images.size(0)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        test_loss /= (total + 1e-12)
        test_acc = correct / (total + 1e-12)

        print(f"[Client {self.cid}] Local eval -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")

        # Return ONLY CNN weights to server
        dense_params = get_parameters_from_model(self._base_model(), state_keys=self.shared_state_keys)
        dense_payload_kb = payload_size_kb(dense_params)
        result_metrics = {
            "accuracy": test_acc,
            "payload_kb": dense_payload_kb,
            "fit_duration_sec": float(time.perf_counter() - round_started_at),
        }

        if use_dp and self.privacy_engine is not None:
            try:
                result_metrics["epsilon"] = float(self.privacy_engine.get_epsilon(delta=target_delta))
            except Exception:
                pass

        if use_sparsification:
            sparse_params, masks = sparsify_parameters(dense_params, k=sparsify_k)
            result_metrics["sparse_payload_kb"] = sparse_payload_size_kb(sparse_params, masks)
            print(
                f"[Client {self.cid}] Payload dense={dense_payload_kb:.2f} KB, sparse={result_metrics['sparse_payload_kb']:.2f} KB"
            )
            return sparse_params, len(self.train_loader.dataset), result_metrics

        print(f"[Client {self.cid}] Payload dense={dense_payload_kb:.2f} KB")
        return dense_params, len(self.train_loader.dataset), result_metrics

    def evaluate(self, parameters, config):
        # Evaluate current global CNN + local head for reporting only
        if parameters is not None and len(parameters) > 0:
            set_parameters_to_model(self._base_model(), parameters, state_keys=self.shared_state_keys)

        self.model.eval()
        self.local_head.eval()

        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.model(images)
                logits = self.local_head(features)

                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=1)

                test_loss += loss.item() * images.size(0)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        test_loss /= (total + 1e-12)
        test_acc = correct / (total + 1e-12)

        return float(test_loss), len(self.test_loader.dataset), {"accuracy": float(test_acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID, e.g., 1")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["simplecnn", "efficientnet_b0"], help="Feature extractor backbone")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for local head (and SimpleCNN backbone if trainable)")
    parser.add_argument("--train_backbone", action=argparse.BooleanOptionalAction, default=True, help="Fine-tune the SHARED backbone (real FedAvg).")
    parser.add_argument("--backbone_lr", type=float, default=None, help="Optional separate LR for backbone fine-tuning (default: 1e-4 for EfficientNet, else --lr)")
    parser.add_argument("--trainable_blocks", type=int, default=1, help="For EfficientNet-B0, unfreeze only the last N feature blocks to reduce latency.")
    parser.add_argument("--max_trainable_blocks", type=int, default=1, help="Maximum EfficientNet feature blocks reserved for communication and progressive unfreezing.")
    parser.add_argument("--address", type=str, default="127.0.0.1:8080", help="gRPC server address")
    args = parser.parse_args()

    client = FlowerClient(
        cid=args.cid,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        train_backbone=args.train_backbone,
        backbone_lr=args.backbone_lr,
        trainable_blocks=args.trainable_blocks,
        max_trainable_blocks=args.max_trainable_blocks,
    )
    fl.client.start_client(server_address=args.address, client=client.to_client())


if __name__ == "__main__":
    main()
