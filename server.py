import argparse
import json
import matplotlib.pyplot as plt
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import ndarrays_to_parameters

from dataset_loader import load_global_test_loader, load_imagefolder_dataloaders
from models import EfficientNetB0Extractor, SimpleCNN
from utils import (
    configure_backbone_training,
    get_device,
    get_parameters_from_model,
    get_trainable_state_keys,
    model_size_kb,
    payload_size_kb,
    set_global_seeds,
    set_parameters_to_model,
)


round_accuracies = []  # RF accuracy collected after each federated round
round_client_accuracies = []  # Weighted client-side eval accuracy after each federated round
latest_client_accuracy: Optional[float] = None


def _save_backbone_checkpoint(model: torch.nn.Module, out_path: str) -> None:
    try:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target)
    except Exception as e:
        print(f"[Server] Failed to save backbone checkpoint to {out_path}: {e}")


def _copy_artifact_if_exists(source_path: str, target_path: str) -> None:
    source = Path(source_path)
    if not source.exists():
        return
    try:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    except Exception as e:
        print(f"[Server] Failed to copy artifact {source_path} -> {target_path}: {e}")


def _parse_progressive_schedule(schedule: str) -> Dict[int, int]:
    parsed: Dict[int, int] = {}
    if not schedule:
        return parsed
    for entry in schedule.split(","):
        item = entry.strip()
        if not item:
            continue
        round_str, blocks_str = item.split(":", 1)
        parsed[int(round_str)] = int(blocks_str)
    return parsed


def _blocks_for_round(default_blocks: int, schedule: Dict[int, int], server_round: int) -> int:
    active_blocks = int(default_blocks)
    for round_idx in sorted(schedule):
        if server_round >= round_idx:
            active_blocks = int(schedule[round_idx])
        else:
            break
    return active_blocks


def _build_metrics_payload(
    server_round: int,
    num_rounds: int,
    num_clients: int,
    preset: str,
    payload_kb: Optional[float],
    quantized_model_kb: Optional[float],
    status: str,
    message: Optional[str] = None,
    rf_evaluated: Optional[bool] = None,
) -> Dict:
    payload: Dict = {
        "last_round": int(server_round),
        "round_num": int(server_round),
        "rounds_expected": int(num_rounds),
        "num_clients": int(num_clients),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backbone": preset,
        "payload_kb": float(payload_kb) if payload_kb is not None else None,
        "quantized_model_kb": float(quantized_model_kb) if quantized_model_kb is not None else None,
        "status": status,
        "training_complete": bool(server_round >= num_rounds and status == "completed"),
    }
    if message:
        payload["message"] = message
    if rf_evaluated is not None:
        payload["rf_evaluated"] = bool(rf_evaluated)
    return payload


def _aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    global latest_client_accuracy
    if not metrics:
        latest_client_accuracy = None
        return {}

    total_examples = sum(int(num_examples) for num_examples, _ in metrics)
    if total_examples <= 0:
        latest_client_accuracy = None
        return {}

    weighted_accuracy = 0.0
    weighted_duration = 0.0
    weighted_payload = 0.0
    for num_examples, metric_dict in metrics:
        weight = float(num_examples) / float(total_examples)
        weighted_accuracy += weight * float(metric_dict.get("accuracy", 0.0))
        weighted_duration += weight * float(metric_dict.get("fit_duration_sec", 0.0))
        weighted_payload += weight * float(metric_dict.get("payload_kb", 0.0))

    latest_client_accuracy = float(weighted_accuracy)
    round_client_accuracies.append(latest_client_accuracy)
    return {
        "client_accuracy": latest_client_accuracy,
        "fit_duration_sec": float(weighted_duration),
        "payload_kb": float(weighted_payload),
    }


def _save_metrics(accuracies, out_path: str = "metrics.json", extra: Optional[Dict] = None):
    """Persist metrics for external UIs (e.g., Streamlit)."""
    payload = {"accuracies": accuracies}
    if extra:
        payload.update(extra)

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[Server] Failed to write {out_path}: {e}")


def _maybe_update_best_metrics(
    accuracies: List[float],
    source_metrics_path: str,
    best_out_path: str = "best_metrics.json",
    extra: Optional[Dict] = None,
) -> None:
    if not accuracies:
        return

    candidate_best = float(max(accuracies))
    existing_best = float("-inf")
    if Path(best_out_path).exists():
        try:
            with open(best_out_path, "r", encoding="utf-8") as handle:
                current_best_payload = json.load(handle)
            existing_best = float(current_best_payload.get("best_accuracy", float("-inf")))
        except Exception:
            existing_best = float("-inf")

    if candidate_best <= existing_best:
        return

    best_payload: Dict = {
        "accuracies": accuracies,
        "best_accuracy": candidate_best,
        "source_metrics_path": str(source_metrics_path),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        best_payload.update(extra)

    try:
        Path(best_out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(best_out_path, "w", encoding="utf-8") as handle:
            json.dump(best_payload, handle, indent=2)
    except Exception as e:
        print(f"[Server] Failed to write {best_out_path}: {e}")


def _build_backbone(backbone: str) -> torch.nn.Module:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return EfficientNetB0Extractor(pretrained=True)
    return SimpleCNN()


def _preset_for_backbone(backbone: str) -> str:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return "efficientnet_b0"
    return "simplecnn"


def _train_and_evaluate_global_rf(
    backbone_model: torch.nn.Module,
    client_loaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    global_test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    rf_out_path: str = "global_rf.pkl",
    pca_out_path: str = "global_pca.pkl",
) -> float:
    """Train a centralized PCA+RandomForest on backbone features and evaluate on global test."""
    try:
        import joblib
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError(
            "Missing sklearn/joblib dependencies for RandomForest evaluation. "
            "Install with: pip install scikit-learn joblib"
        ) from e

    backbone_model.eval()

    X_train, y_train = [], []
    with torch.no_grad():
        for train_loader, _ in client_loaders:
            for images, labels in train_loader:
                images = images.to(device)
                feats = backbone_model(images)
                X_train.append(feats.cpu().numpy())
                y_train.append(labels.numpy())

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    pca = PCA(n_components=0.90, whiten=True)
    X_train_pca = pca.fit_transform(X_train)

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

    # Evaluate on global test
    X_test, y_test = [], []
    with torch.no_grad():
        for images, labels in global_test_loader:
            images = images.to(device)
            feats = backbone_model(images)
            X_test.append(feats.cpu().numpy())
            y_test.append(labels.numpy())

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    X_test_pca = pca.transform(X_test)

    acc = float(rf.score(X_test_pca, y_test))

    # Persist models for downstream inference
    joblib.dump(rf, rf_out_path)
    joblib.dump(pca, pca_out_path)

    return acc


def get_evaluate_fn(
    model: torch.nn.Module,
    global_test_loader,
    device: torch.device,
    client_loaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    shared_state_keys: List[str],
    preset: str,
    rf_eval_interval: int,
    num_rounds: int,
    log_diagnostics: bool,
    run_initial_eval: bool,
    metrics_out: str,
    best_metrics_out: str,
    run_artifact_dir: str,
    best_artifact_dir: str,
):
    rf_eval_interval = max(1, int(rf_eval_interval))
    last_rf_acc: Optional[float] = None
    best_run_rf_acc: float = float("-inf")

    run_artifact_dir_path = Path(run_artifact_dir)
    best_artifact_dir_path = Path(best_artifact_dir)
    run_rf_path = str(run_artifact_dir_path / "global_rf.pkl")
    run_pca_path = str(run_artifact_dir_path / "global_pca.pkl")
    run_backbone_latest_path = str(run_artifact_dir_path / "global_backbone_latest.pt")
    run_backbone_best_path = str(run_artifact_dir_path / "global_backbone_best_in_run.pt")
    run_backbone_final_path = str(run_artifact_dir_path / "global_backbone_final.pt")
    run_metadata_path = str(run_artifact_dir_path / "run_summary.json")
    best_backbone_path = str(best_artifact_dir_path / "global_backbone_best.pt")
    best_rf_path = str(best_artifact_dir_path / "global_rf_best.pkl")
    best_pca_path = str(best_artifact_dir_path / "global_pca_best.pkl")
    best_metadata_path = str(best_artifact_dir_path / "best_artifacts.json")

    def evaluate(server_round, parameters, config):
        nonlocal last_rf_acc, best_run_rf_acc
        set_global_seeds(42)
        payload_kb = None
        quantized_model_kb = None
        # Load aggregated backbone weights
        if parameters is not None:
            if hasattr(parameters, "tensors"):
                ndarrays = fl.common.parameters_to_ndarrays(parameters)
            else:
                ndarrays = parameters
            set_parameters_to_model(model, ndarrays, state_keys=shared_state_keys)
            payload_kb = payload_size_kb(ndarrays)
            if log_diagnostics:
                original_model_kb = model_size_kb(model)
                try:
                    quantized_model = torch.quantization.quantize_dynamic(
                        model.cpu(),
                        {nn.Linear, nn.Conv2d},
                        dtype=torch.qint8,
                    )
                    quantized_model_kb = model_size_kb(quantized_model)
                    model.to(device)
                except Exception as exc:
                    quantized_model_kb = original_model_kb
                    print(f"[Server] Quantization fallback: {exc}")
                print(
                    f"[Server] Round {server_round}: payload={payload_kb:.2f} KB | model={original_model_kb:.2f} KB | quantized={quantized_model_kb:.2f} KB"
                )
            else:
                print(f"[Server] Round {server_round}: payload={payload_kb:.2f} KB")
            _save_backbone_checkpoint(model, run_backbone_latest_path)

        if server_round == 0 and not run_initial_eval:
            print("[Server] Round 0: skipping expensive initial RF evaluation to start federated training faster")
            _save_metrics(
                round_accuracies,
                out_path=metrics_out,
                extra=_build_metrics_payload(
                    server_round=server_round,
                    num_rounds=num_rounds,
                    num_clients=len(client_loaders),
                    preset=preset,
                    payload_kb=payload_kb,
                    quantized_model_kb=quantized_model_kb,
                    status="started",
                    message="Initial evaluation skipped to reduce startup latency.",
                    rf_evaluated=False,
                ),
            )
            return None

        should_run_rf = (
            last_rf_acc is None
            or server_round == num_rounds
            or (server_round % rf_eval_interval == 0)
        )
        if should_run_rf:
            rf_acc = _train_and_evaluate_global_rf(
                model,
                client_loaders=client_loaders,
                global_test_loader=global_test_loader,
                device=device,
                rf_out_path=run_rf_path,
                pca_out_path=run_pca_path,
            )
            last_rf_acc = rf_acc
            rf_evaluated = True
        else:
            rf_acc = last_rf_acc
            rf_evaluated = False
            print(
                f"[Server] Round {server_round}: skipped RF retraining to reduce latency; "
                f"reusing last RF acc={rf_acc:.4f}"
            )

        round_accuracies.append(rf_acc)
        if rf_evaluated and rf_acc > best_run_rf_acc:
            best_run_rf_acc = rf_acc
            _save_backbone_checkpoint(model, run_backbone_best_path)

        _save_metrics(
            round_client_accuracies if round_client_accuracies else round_accuracies,
            out_path=metrics_out,
            extra={
                **_build_metrics_payload(
                    server_round=server_round,
                    num_rounds=num_rounds,
                    num_clients=len(client_loaders),
                    preset=preset,
                    payload_kb=payload_kb,
                    quantized_model_kb=quantized_model_kb,
                    status="completed" if server_round >= num_rounds else "running",
                    message=(
                        "RF evaluation completed for this round."
                        if rf_evaluated
                        else "RF evaluation skipped for this round; last measured score reused."
                    ),
                    rf_evaluated=rf_evaluated,
                ),
                "client_accuracies": round_client_accuracies,
                "rf_accuracies": round_accuracies,
                "latest_client_accuracy": latest_client_accuracy,
            },
        )
        _maybe_update_best_metrics(
            round_accuracies,
            source_metrics_path=metrics_out,
            best_out_path=best_metrics_out,
            extra={
                "last_round": int(server_round),
                "round_num": int(server_round),
                "rounds_expected": int(num_rounds),
                "num_clients": int(len(client_loaders)),
                "backbone": preset,
                "status": "completed" if server_round >= num_rounds else "running",
                "client_accuracies": round_client_accuracies,
                "rf_accuracies": round_accuracies,
            },
        )
        if server_round >= num_rounds:
            _save_backbone_checkpoint(model, run_backbone_final_path)
            run_summary = {
                "run_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "best_rf_accuracy": float(best_run_rf_acc if best_run_rf_acc != float("-inf") else rf_acc),
                "final_rf_accuracy": float(rf_acc),
                "final_client_accuracy": float(latest_client_accuracy) if latest_client_accuracy is not None else None,
                "best_backbone_path": run_backbone_best_path if Path(run_backbone_best_path).exists() else None,
                "final_backbone_path": run_backbone_final_path,
                "rf_path": run_rf_path if Path(run_rf_path).exists() else None,
                "pca_path": run_pca_path if Path(run_pca_path).exists() else None,
                "metrics_path": metrics_out,
                "selection_metric": "best_rf_accuracy",
            }
            _save_metrics([], out_path=run_metadata_path, extra=run_summary)

            existing_best = float("-inf")
            if Path(best_metrics_out).exists():
                try:
                    with open(best_metrics_out, "r", encoding="utf-8") as handle:
                        best_payload = json.load(handle)
                    existing_best = float(best_payload.get("best_accuracy", float("-inf")))
                except Exception:
                    existing_best = float("-inf")

            if float(best_run_rf_acc if best_run_rf_acc != float("-inf") else rf_acc) >= existing_best:
                _copy_artifact_if_exists(run_backbone_best_path, best_backbone_path)
                _copy_artifact_if_exists(run_rf_path, best_rf_path)
                _copy_artifact_if_exists(run_pca_path, best_pca_path)
                _save_metrics(
                    [],
                    out_path=best_metadata_path,
                    extra={
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "selection_metric": "best_rf_accuracy",
                        "best_rf_accuracy": float(best_run_rf_acc if best_run_rf_acc != float("-inf") else rf_acc),
                        "source_metrics_path": metrics_out,
                        "source_backbone_path": run_backbone_best_path if Path(run_backbone_best_path).exists() else run_backbone_final_path,
                        "source_rf_path": run_rf_path if Path(run_rf_path).exists() else None,
                        "source_pca_path": run_pca_path if Path(run_pca_path).exists() else None,
                        "best_backbone_path": best_backbone_path,
                        "best_rf_path": best_rf_path if Path(best_rf_path).exists() else None,
                        "best_pca_path": best_pca_path if Path(best_pca_path).exists() else None,
                    },
                )

        print(f"[Server] Round {server_round}: GLOBAL RF acc (on global_test) = {rf_acc:.4f}")
        # Flower expects (loss, metrics). We use 1-acc as a pseudo-loss.
        return float(1.0 - rf_acc), {"rf_accuracy": float(rf_acc)}

    return evaluate


def get_on_fit_config_fn(
    epochs: int,
    batch_size: int,
    max_batches_per_round: int,
    default_trainable_blocks: int,
    progressive_schedule: Dict[int, int],
):
    def on_fit_config_fn(server_round: int):
        active_blocks = _blocks_for_round(default_trainable_blocks, progressive_schedule, server_round)
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "max_batches_per_round": max_batches_per_round,
            "trainable_blocks": active_blocks,
            "use_sparsification": False,
            "sparsify_k": 0.01,
            "use_dp": False,
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
            "max_grad_norm": 1.2,
            "use_fedprox": False,
            "fedprox_mu": 0.01,
        }

    return on_fit_config_fn


def plot_accuracies(accuracies, out_path: str = "accuracy_curve.png"):
    if not accuracies:
        return
    rounds = list(range(1, len(accuracies) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(rounds, accuracies, marker="o")
    plt.title("Global RF Accuracy vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Server] Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["simplecnn", "efficientnet_b0"], help="Feature extractor backbone")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients (expects data/client_1..data/client_N)")
    parser.add_argument("--client_selection", type=str, default="all", choices=["all", "sampled"], help="Use all connected clients every round, or allow sampled participation for production-style scaling.")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for clients and server RF eval")
    parser.add_argument("--max_batches_per_round", type=int, default=0, help="Limit each client to this many shuffled batches per round. 0 uses the full local loader.")
    parser.add_argument("--address", type=str, default="127.0.0.1:8080", help="gRPC server address")
    parser.add_argument("--metrics_out", type=str, default="metrics.json", help="Path to the metrics JSON written for this run.")
    parser.add_argument("--best_metrics_out", type=str, default="best_metrics.json", help="Path to the cross-run best metrics snapshot.")
    parser.add_argument("--rf_eval_interval", type=int, default=2, help="Run the expensive server-side PCA+RF evaluation every N rounds and on the final round.")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of available clients sampled each round.")
    parser.add_argument("--min_fit_clients", type=int, default=2, help="Minimum number of clients to train in each round.")
    parser.add_argument("--min_available_clients", type=int, default=3, help="Minimum number of connected clients before training starts.")
    parser.add_argument("--run_artifact_dir", type=str, default="runs/latest_artifacts", help="Directory for run-specific checkpoints and RF/PCA artifacts.")
    parser.add_argument("--best_artifact_dir", type=str, default="best_artifacts", help="Directory holding the best preserved checkpoints/artifacts across runs.")
    parser.add_argument("--log_diagnostics", action=argparse.BooleanOptionalAction, default=False, help="Enable expensive per-round model size/quantization diagnostics.")
    parser.add_argument("--initial_eval", action=argparse.BooleanOptionalAction, default=False, help="Run the expensive server-side evaluation before round 1 starts.")
    parser.add_argument("--trainable_blocks", type=int, default=1, help="For EfficientNet-B0, federate only the last N feature blocks.")
    parser.add_argument("--max_trainable_blocks", type=int, default=0, help="Maximum EfficientNet feature blocks reserved for communication; 0 derives from trainable blocks and any progressive schedule.")
    parser.add_argument("--progressive_unfreeze_schedule", type=str, default="", help="Comma-separated round:block schedule, e.g. '1:1,3:2,5:3'.")
    args = parser.parse_args()
    set_global_seeds(42)
    progressive_schedule = _parse_progressive_schedule(args.progressive_unfreeze_schedule)
    args.trainable_blocks = _blocks_for_round(args.trainable_blocks, progressive_schedule, 1)
    max_schedule_blocks = max(progressive_schedule.values(), default=args.trainable_blocks)
    args.max_trainable_blocks = int(args.max_trainable_blocks or max(args.trainable_blocks, max_schedule_blocks))
    if args.client_selection == "all":
        args.fraction_fit = 1.0
        args.min_fit_clients = int(args.num_clients)
        args.min_available_clients = int(args.num_clients)
    else:
        args.min_fit_clients = min(int(args.min_fit_clients), int(args.num_clients))
        args.min_available_clients = min(int(args.min_available_clients), int(args.num_clients))

    print(
        f"[Server] client_selection={args.client_selection} | fraction_fit={args.fraction_fit:.2f} | "
        f"min_fit_clients={args.min_fit_clients} | min_available_clients={args.min_available_clients} | "
        f"trainable_blocks={args.trainable_blocks} | max_trainable_blocks={args.max_trainable_blocks} | "
        f"progressive_schedule={args.progressive_unfreeze_schedule or 'off'} | metrics_out={args.metrics_out} | "
        f"best_metrics_out={args.best_metrics_out} | run_artifact_dir={args.run_artifact_dir} | "
        f"best_artifact_dir={args.best_artifact_dir}"
    )

    device = get_device()
    preset = _preset_for_backbone(args.backbone)

    # Global test loader for RF evaluation (uses same preset as backbone)
    global_test_loader, _num_classes = load_global_test_loader(
        "data/global_test",
        batch_size=args.batch_size,
        preset=preset,
    )

    client_loaders = []
    for cid in range(1, int(args.num_clients) + 1):
        train_dir = f"data/client_{cid}/train"
        test_dir = f"data/client_{cid}/test"
        train_loader, test_loader, _ = load_imagefolder_dataloaders(
            train_dir,
            test_dir,
            batch_size=args.batch_size,
            preset=preset,
        )
        client_loaders.append((train_loader, test_loader))

    # Backbone model (for evaluation and initial parameters)
    model = _build_backbone(args.backbone).to(device)
    configure_backbone_training(
        model,
        backbone=args.backbone,
        train_backbone=True,
        trainable_blocks=args.max_trainable_blocks,
    )
    shared_state_keys = get_trainable_state_keys(model)

    # Initial parameters so all clients start from same weights
    initial_ndarrays = get_parameters_from_model(model, state_keys=shared_state_keys)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(
            model,
            global_test_loader=global_test_loader,
            device=device,
            client_loaders=client_loaders,
            shared_state_keys=shared_state_keys,
            preset=preset,
            rf_eval_interval=args.rf_eval_interval,
            num_rounds=args.num_rounds,
            log_diagnostics=args.log_diagnostics,
            run_initial_eval=args.initial_eval,
            metrics_out=args.metrics_out,
            best_metrics_out=args.best_metrics_out,
            run_artifact_dir=args.run_artifact_dir,
            best_artifact_dir=args.best_artifact_dir,
        ),
        on_fit_config_fn=get_on_fit_config_fn(
            args.epochs,
            args.batch_size,
            args.max_batches_per_round,
            args.trainable_blocks,
            progressive_schedule,
        ),
        fit_metrics_aggregation_fn=_aggregate_fit_metrics,
        initial_parameters=initial_parameters,
        fraction_fit=float(args.fraction_fit),
        min_fit_clients=int(args.min_fit_clients),
        min_available_clients=int(args.min_available_clients),
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    plot_accuracies(round_accuracies, out_path="accuracy_curve.png")


if __name__ == "__main__":
    main()
