# FedDisaster: Federated Multi-Class Disaster Classification (Flower + PyTorch)

This is a working federated learning system for multi-class disaster image classification (including flood and non-flood categories) using Flower (flwr) and PyTorch.
It supports offline local datasets per client, a held-out global test set, and round-wise server evaluation.

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
- utils.py                          ... Shared utilities (parameter conversion, device selection)
- requirements.txt                  ... project dependencies

Assumptions:
- Data uses `torchvision.datasets.ImageFolder` format.
- Every client and `data/global_test` should use the same class folder names.
- CPU-first workflow; `EfficientNet-B0` is supported but heavier than `SimpleCNN`.

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
- `python -m venv .venv`
- `.venv\Scripts\activate`  (PowerShell on Windows)
- `pip install -r requirements.txt`

3) Run server (terminal 1)
- Default backbone (`simplecnn`):
  - `python server.py --num_rounds 5 --epochs 1 --batch_size 32`
- EfficientNet-B0 backbone:
  - `python server.py --backbone efficientnet_b0 --num_rounds 5 --epochs 1 --batch_size 32`

4) Run clients (separate terminals)
- Default backbone (`simplecnn`):
  - `python client.py --cid 1`
  - `python client.py --cid 2`
  - `python client.py --cid 3`
- EfficientNet-B0 backbone:
  - `python client.py --cid 1 --backbone efficientnet_b0`
  - `python client.py --cid 2 --backbone efficientnet_b0`
  - `python client.py --cid 3 --backbone efficientnet_b0`

5) Convenience scripts (PowerShell)
- `scripts/start_server.ps1 -Rounds 5 -Epochs 1 -BatchSize 32`
- `scripts/start_clients.ps1 -Count 3 -BatchSize 32`

Outputs and metrics:
- `metrics.json` (round-wise metrics, backbone, timestamps)
- `accuracy_curve.png` (accuracy vs federated round)
- `global_rf.pkl` and `global_pca.pkl` (server-side RF/PCA artifacts)
- `global_cnn.pt` (saved backbone weights from demo flow)

Notes:
- Number of classes is detected dynamically from folder structure.
- FedAvg exchanges backbone parameters only; raw client images never leave local client folders.
- Class mapping consistency is validated across clients/global test in the demo workflow.
