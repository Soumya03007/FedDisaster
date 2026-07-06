# FedDisaster: Federated Multi-Class Disaster Classification (Flower + PyTorch)

This is a working federated learning system for multi-class disaster image classification using Flower (flwr), PyTorch, EfficientNet-B0 feature extraction, and a server-side PCA + RandomForest evaluator.
It supports offline local datasets per client, a held-out global test set, and round-wise server evaluation without exchanging raw client images.

Current verified result:
- Global test samples: 1,353
- Classes: 6
- Best global PCA+RandomForest accuracy: 94.0872%
- Best weighted client-side accuracy: 95.2243%
- Details: see `RESULTS.md`

Quick artifact verification:

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

Expected output:

```text
classes=6
samples=1353
accuracy=0.940872
```

Project structure:
- data/
  - client_1/train/ ... images organized by class folders
  - client_1/test/  ... images organized by class folders
  - client_2/train/ ...
  - client_2/test/  ...
  - global_test/    ... images organized by class folders
- client.py                         ... Flower NumPyClient implementation (local train/eval + optional backbone fine-tuning)
- server.py                         ... FedAvg server + round-wise global RF evaluation + metrics persistence
- dataset_loader.py                 ... Robust ImageFolder data loaders, class counting, and presets
- models.py                         ... `SimpleCNN` and `EfficientNetB0Extractor` backbones + local head
- data/setup_multiclass_dataset.py  ... Multi-source multi-class dataset setup/distribution utility
- simple_demo.py                    ... End-to-end local federated simulation workflow
- scripts/evaluate_best_artifacts.py ... Reproducibility check for saved EfficientNet + RF artifacts
- utils.py                          ... Shared utilities (parameter conversion, device selection)
- requirements.txt                  ... project dependencies

Assumptions:
- Data uses `torchvision.datasets.ImageFolder` format.
- Every client and `data/global_test` should use the same class folder names.
- `EfficientNet-B0` is the default federated backbone; to reduce latency, the repo now fine-tunes only the last EfficientNet feature block by default.

1) Prepare offline data
- Put your client-specific datasets here:
  - `data/client_1/train/<class_name>/*.jpg|png`
  - `data/client_1/test/<class_name>/*.jpg|png`
  - `data/client_2/train/<class_name>/*.jpg|png`
  - `data/client_2/test/<class_name>/*.jpg|png`
  - ...
- Put the held-out global test set here:
  - `data/global_test/<class_name>/*.jpg|png`

Optional multi-class setup helper:
- `python data/setup_multiclass_dataset.py --disaster_sources flood=data/_organized fire=path/to/fire landslide=path/to/landslide --target_root data --num_clients 3 --force`

2) Install dependencies (recommended in a virtual environment)
- Linux/macOS:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Windows PowerShell:
  - `python -m venv .venv`
  - `.venv\Scripts\activate`
  - `pip install -r requirements.txt`

3) Verify saved artifacts
- `python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64`
- Expected accuracy: `0.940872`

4) Run server (terminal 1)
- Default low-latency EfficientNet path:
  - `python server.py --backbone efficientnet_b0 --num_rounds 5 --epochs 1 --batch_size 32 --rf_eval_interval 2`
- If you want every client in every round:
  - `python server.py --backbone efficientnet_b0 --num_rounds 5 --epochs 1 --batch_size 32 --fraction_fit 1.0`

5) Run clients (separate terminals)
- Default federated EfficientNet path:
  - `python client.py --cid 1 --backbone efficientnet_b0 --train_backbone --trainable_blocks 1`
  - `python client.py --cid 2 --backbone efficientnet_b0 --train_backbone --trainable_blocks 1`
  - `python client.py --cid 3 --backbone efficientnet_b0 --train_backbone --trainable_blocks 1`
- To fine-tune a larger shared slice of EfficientNet:
  - increase `--trainable_blocks` from `1` to `2` or `3`

6) Convenience scripts (PowerShell)
- `scripts/start_server.ps1 -Rounds 5 -Epochs 1 -BatchSize 32 -RfEvalInterval 2`
- `scripts/start_clients.ps1 -Count 3 -BatchSize 32 -TrainableBlocks 1`
- One-command launcher:
  - `scripts/run_federated.ps1 -NumClients 3 -NumRounds 5 -Epochs 1 -BatchSize 32 -TrainableBlocks 1`
  - or `python scripts/run_federated.py --num_clients 3 --num_rounds 5 --epochs 1 --batch_size 32 --trainable_blocks 1`
  - default behavior is `client_selection=all`, which makes all launched clients train every round
  - for production-style sampling, use `-ClientSelection sampled -FractionFit 0.66` or `--client_selection sampled --fraction_fit 0.66`
  - for faster and more varied local updates, use `-MaxBatchesPerRound 20` or `--max_batches_per_round 20`

Quick demo note:
- `python simple_demo.py` now defaults to `--backbone efficientnet_b0`.
- `simplecnn` is still available with `python simple_demo.py --backbone simplecnn`, but it is kept mainly for backward compatibility.

Outputs and metrics:
- `runs/metrics_YYYYMMDD_HHMMSS.json` (one metrics file per run)
- `runs/artifacts_YYYYMMDD_HHMMSS/` (run-specific backbone, RF, PCA, and run summary)
- `latest_metrics_path.txt` (pointer used by the dashboard to follow the newest run)
- `best_metrics.json` (best-performing run snapshot across all runs)
- `best_artifacts/` (best preserved backbone, RF, PCA, and artifact metadata across runs)
- `accuracy_curve.png` (accuracy vs federated round)
- `global_rf.pkl`, `global_pca.pkl`, and `global_cnn.pt` remain part of the demo flow

Notes:
- Number of classes is detected dynamically from folder structure.
- FedAvg exchanges backbone parameters only; raw client images never leave local client folders.
- Server-side PCA + RandomForest evaluation is intentionally throttled with `--rf_eval_interval` so heavy centralized evaluation does not block every round.
- The one-command launcher skips the expensive initial evaluation and defaults `rf_eval_interval=0`, which means the heavy RF step runs only on the final round.
- Small-team/dev default: `client_selection=all` for easier debugging and reproducibility.
- Production-scale option: `client_selection=sampled` for lower latency and better straggler tolerance.
- `max_batches_per_round` lets each client train on a different shuffled subset of local data each round without permanently splitting the dataset.
- Class mapping consistency is validated across clients/global test in the demo workflow.
