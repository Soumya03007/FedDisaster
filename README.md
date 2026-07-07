# FedDisaster

Federated multi-class disaster image classification with Flower, PyTorch EfficientNet-B0, and verified PCA + RandomForest artifacts.

[![CI](https://github.com/Soumya03007/FedDisaster/actions/workflows/ci.yml/badge.svg)](https://github.com/Soumya03007/FedDisaster/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v0.1.0--efficientnet--rf-blue)](https://github.com/Soumya03007/FedDisaster/releases/tag/v0.1.0-efficientnet-rf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

FedDisaster is a working federated learning pipeline for disaster-scene image classification. It keeps image data local to clients, federates an EfficientNet-B0 backbone with Flower FedAvg, and evaluates global performance with a server-side PCA + RandomForest classifier trained on extracted features.

The project is designed to be inspectable and reproducible: verified artifacts are published in GitHub Releases, the result can be checked with one command, and the repository includes a model card, artifact guide, dataset guide, contribution guide, and lightweight CI.

## Verified Result

| Metric | Value |
| --- | --- |
| Classes | 6 |
| Global test samples | 1,353 |
| Best global PCA+RandomForest accuracy | 94.0872% |
| Best weighted client-side accuracy | 95.2243% |
| Backbone | EfficientNet-B0 |
| Federated strategy | Flower FedAvg |
| Global evaluator | PCA + RandomForest |

Verified release:

```text
https://github.com/Soumya03007/FedDisaster/releases/tag/v0.1.0-efficientnet-rf
```

Detailed result notes are in `RESULTS.md`.

## Why This Project Exists

Disaster image datasets are often fragmented across sources, organizations, or devices. A federated setup lets each client train locally while sharing model updates instead of raw images. FedDisaster explores that workflow for multi-class disaster classification, with practical scripts for:

- local client training,
- federated server orchestration,
- EfficientNet-B0 feature extraction,
- global PCA + RandomForest evaluation,
- verified artifact reuse,
- single-image inference.

## Quickstart

Clone the repository:

```bash
git clone https://github.com/Soumya03007/FedDisaster.git
cd FedDisaster
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Verify the saved artifacts:

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

Expected output:

```text
classes=6
samples=1353
accuracy=0.940872
```

Run inference on one image:

```bash
python predict.py --image path/to/image.jpg --artifacts release
```

Example output:

```json
{
  "prediction": "Damaged_Infrastructure",
  "class_index": 0,
  "top_k": [
    {
      "label": "Damaged_Infrastructure",
      "probability": 0.4079
    }
  ]
}
```

## Dataset Format

FedDisaster uses `torchvision.datasets.ImageFolder` layout. Every client and the global test set must use the same class folder names.

```text
data/
  client_1/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  client_2/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  client_3/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  global_test/
    <class_name>/*.jpg
```

Verified classes:

```text
Damaged_Infrastructure
Fire_Disaster
Human_Damage
Land_Disaster
Non_Damage
Water_Disaster
```

See `DATASET.md` for sample dataset guidance and preparation notes.

## Architecture

```text
Client image folders
        |
        v
Local client training
EfficientNet-B0 backbone + private local head
        |
        v
Flower FedAvg
Shared backbone parameter aggregation
        |
        v
Server-side evaluation
EfficientNet features -> PCA -> RandomForest
        |
        v
Global metrics and preserved artifacts
```

Important details:

- Raw client images are not sent to the server.
- Clients train a local classification head that is not shared.
- FedAvg exchanges selected EfficientNet-B0 backbone parameters.
- The server trains/evaluates a PCA + RandomForest classifier over extracted features.
- Heavy RF evaluation can be throttled with `--rf_eval_interval`.

## Running Federated Training

Start the server:

```bash
python server.py --backbone efficientnet_b0 --num_rounds 5 --epochs 1 --batch_size 32 --rf_eval_interval 2
```

Start clients in separate terminals:

```bash
python client.py --cid 1 --backbone efficientnet_b0 --train_backbone --trainable_blocks 1
python client.py --cid 2 --backbone efficientnet_b0 --train_backbone --trainable_blocks 1
python client.py --cid 3 --backbone efficientnet_b0 --train_backbone --trainable_blocks 1
```

Use all clients every round:

```bash
python server.py --backbone efficientnet_b0 --num_rounds 5 --epochs 1 --batch_size 32 --fraction_fit 1.0
```

Fine-tune a larger EfficientNet slice by increasing:

```bash
--trainable_blocks 2
```

or:

```bash
--trainable_blocks 3
```

## Local Simulation

Run the in-process demo:

```bash
python simple_demo.py --backbone efficientnet_b0 --num_rounds 5 --epochs 1 --batch_size 32
```

`simplecnn` remains available for smoke tests:

```bash
python simple_demo.py --backbone simplecnn
```

The reported 94.0872% result is from EfficientNet-B0 artifacts, not the SimpleCNN compatibility path.

## PowerShell Convenience Scripts

```powershell
scripts/start_server.ps1 -Rounds 5 -Epochs 1 -BatchSize 32 -RfEvalInterval 2
scripts/start_clients.ps1 -Count 3 -BatchSize 32 -TrainableBlocks 1
scripts/run_federated.ps1 -NumClients 3 -NumRounds 5 -Epochs 1 -BatchSize 32 -TrainableBlocks 1
```

Python launcher:

```bash
python scripts/run_federated.py --num_clients 3 --num_rounds 5 --epochs 1 --batch_size 32 --trainable_blocks 1
```

## Artifacts

Current verified artifacts:

```text
global_cnn.pt
global_pca.pkl
global_rf.pkl
best_artifacts/global_backbone_best.pt
best_artifacts/global_pca_best.pkl
best_artifacts/global_rf_best.pkl
```

Artifact release:

```text
https://github.com/Soumya03007/FedDisaster/releases/tag/v0.1.0-efficientnet-rf
```

See `ARTIFACTS.md` for what each artifact does, how it should be packaged, and how maintainers should update it.

## Repository Map

| Path | Purpose |
| --- | --- |
| `client.py` | Flower NumPyClient implementation |
| `server.py` | FedAvg server, RF evaluation, metrics persistence |
| `models.py` | SimpleCNN, EfficientNetB0Extractor, LocalHead |
| `dataset_loader.py` | Robust ImageFolder loaders and transforms |
| `predict.py` | Single-image inference with saved artifacts |
| `simple_demo.py` | In-process federated simulation |
| `scripts/evaluate_best_artifacts.py` | Reproduce the verified artifact accuracy |
| `data/setup_multiclass_dataset.py` | Multi-source dataset distribution helper |
| `streamlit_app.py` | Metrics/dashboard viewer |
| `RESULTS.md` | Verified metrics and historical run summary |
| `MODEL_CARD.md` | Intended use, limitations, and model details |
| `ARTIFACTS.md` | Release artifact documentation |
| `DATASET.md` | Dataset layout and preparation guide |
| `CONTRIBUTING.md` | Contributor workflow |

## Metrics And Outputs

Typical outputs:

```text
runs/metrics_YYYYMMDD_HHMMSS.json
runs/artifacts_YYYYMMDD_HHMMSS/
latest_metrics_path.txt
best_metrics.json
best_artifacts/
accuracy_curve.png
global_cnn.pt
global_pca.pkl
global_rf.pkl
```

The dashboard reads metrics files and can track current/best runs.

## Collaboration

Start with the collaboration docs:

```text
CONTRIBUTING.md
DATASET.md
MODEL_CARD.md
ARTIFACTS.md
RESULTS.md
```

Use a feature branch for every change:

```bash
git checkout main
git pull origin main
git checkout -b feature/<short-description>
```

Examples:

```bash
git checkout -b feature/batch-inference
git checkout -b docs/dataset-guide
git checkout -b experiment/convnext-backbone
git checkout -b fix/client-class-mapping
```

Make focused changes, then run local checks:

```bash
python -m compileall client.py server.py dataset_loader.py models.py simple_demo.py utils.py predict.py scripts
python -m json.tool metrics.json
python -m json.tool best_metrics.json
python -m json.tool best_artifacts/best_artifacts.json
```

If your change touches inference or artifacts, verify the saved result:

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

If your change touches `predict.py`, test one image:

```bash
python predict.py --image path/to/image.jpg --artifacts release --top_k 3
```

Review your diff before committing:

```bash
git status --short
git diff
git diff --stat
```

Stage only related files:

```bash
git add path/to/changed_file.py path/to/changed_doc.md
```

Commit with a clear message:

```bash
git commit -m "Add batch inference for artifact classifier"
```

Keep your branch updated before opening a pull request:

```bash
git fetch origin
git rebase origin/main
```

Push the branch:

```bash
git push -u origin feature/<short-description>
```

Open a pull request with:

- what changed,
- why it changed,
- commands used to test it,
- metric impact, if any,
- artifact changes, if any.

Pull request body template:

```markdown
## Summary
- 

## Validation
- [ ] `python -m compileall client.py server.py dataset_loader.py models.py simple_demo.py utils.py predict.py scripts`
- [ ] `python -m json.tool metrics.json`
- [ ] `python -m json.tool best_metrics.json`
- [ ] `python -m json.tool best_artifacts/best_artifacts.json`

## Metric / Artifact Impact
- 

## Notes
- 
```

Avoid broad commits:

```bash
# Avoid this unless you have reviewed every file:
git add .
```

Prefer:

```bash
git add README.md predict.py
```

Good contribution areas:

- add dataset importers,
- improve federated client sampling,
- add per-class metrics and confusion matrices,
- add a small public sample dataset,
- add artifact download automation,
- add inference batch mode,
- improve the Streamlit dashboard,
- benchmark more backbones,
- add privacy/security experiments such as DP or secure aggregation.

If you update model artifacts, also update:

```text
RESULTS.md
MODEL_CARD.md
ARTIFACTS.md
best_metrics.json
best_artifacts/best_artifacts.json
```

## Roadmap

- Batch inference mode for folders of images.
- Optional automatic artifact download from GitHub Releases.
- Confusion matrix and per-class precision/recall reports.
- Tiny public sample dataset for CI-friendly inference.
- Stronger artifact versioning and checksum validation.
- Expanded model comparison: EfficientNet variants, ConvNeXt, ViT, MobileNet.
- Federated privacy experiments with DP and secure aggregation.

## Limitations

- The verified score depends on the saved dataset split and class mapping.
- The global RandomForest is centralized over extracted features; it is not itself federated.
- Saved `.pkl` files should only be loaded from trusted sources.
- Disaster image datasets may contain geographic, source, and event-type bias.
- This project is not a drop-in emergency response system without domain validation and monitoring.

## License

MIT. See `LICENSE`.

## Citation

If you use this project in research or a portfolio write-up, cite the repository and include the verified artifact release:

```text
FedDisaster: Federated Multi-Class Disaster Classification with Flower, PyTorch EfficientNet-B0, and PCA+RandomForest evaluation.
https://github.com/Soumya03007/FedDisaster
https://github.com/Soumya03007/FedDisaster/releases/tag/v0.1.0-efficientnet-rf
```
