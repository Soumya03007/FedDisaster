# Contributing

Thanks for taking an interest in FedDisaster. This project is intended to be reproducible, useful, and straightforward to inspect. Contributions are welcome when they keep that standard intact.

## Ways To Contribute

Good pull requests are focused. Prefer one clear scope:

- model or training improvement,
- dataset preparation improvement,
- documentation update,
- experiment/result update,
- inference workflow improvement,
- dashboard or metrics update,
- CI/reproducibility improvement.

Avoid mixing large artifact changes with unrelated formatting or refactors.

## Development Setup

Clone and enter the repository:

```bash
git clone https://github.com/Soumya03007/FedDisaster.git
cd FedDisaster
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If your system uses the Python launcher on Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Verify The Saved Result

Run this after setup to confirm the artifact path works.

### Linux / macOS

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

### Windows PowerShell

```powershell
python scripts\evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

Expected result:

```text
classes=6
samples=1353
accuracy=0.940872
```

## Run Single-Image Inference

### Linux / macOS

```bash
python predict.py --image path/to/image.jpg --artifacts release --top_k 3
```

### Windows PowerShell

```powershell
python predict.py --image path\to\image.jpg --artifacts release --top_k 3
```

## Branch Workflow

Create a branch for every contribution.

### Linux / macOS

```bash
git checkout main
git pull origin main
git checkout -b feature/<short-description>
```

### Windows PowerShell

```powershell
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

## Local Checks

Run these before opening a pull request.

### Linux / macOS

```bash
python -m compileall client.py server.py dataset_loader.py models.py simple_demo.py utils.py predict.py scripts
python -m json.tool metrics.json
python -m json.tool best_metrics.json
python -m json.tool best_artifacts/best_artifacts.json
```

### Windows PowerShell

```powershell
python -m compileall client.py server.py dataset_loader.py models.py simple_demo.py utils.py predict.py scripts
python -m json.tool metrics.json
python -m json.tool best_metrics.json
python -m json.tool best_artifacts\best_artifacts.json
```

If your change affects training, evaluation, inference, or artifacts, also include the exact command you ran and the output metric.

## Reviewing Your Changes

Before committing:

```bash
git status --short
git diff
git diff --stat
```

Stage only related files:

```bash
git add path/to/changed_file.py path/to/changed_doc.md
```

Avoid broad staging unless you have reviewed every file:

```bash
# Avoid this for routine PRs:
git add .
```

Commit with a clear message:

```bash
git commit -m "Add batch inference for artifact classifier"
```

Keep your branch current:

```bash
git fetch origin
git rebase origin/main
```

Push your branch:

```bash
git push -u origin feature/<short-description>
```

## Pull Request Checklist

Your pull request should include:

- what changed,
- why it changed,
- commands used for validation,
- final metric output if relevant,
- dataset split used if relevant,
- artifact changes if relevant.

Suggested PR body:

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

## Repository Hygiene

- Do not commit local virtual environments such as `.venv/` or `.venv-linux/`.
- Do not commit private or full raw datasets.
- Do not use `git add .` when generated artifacts or run outputs changed unexpectedly.
- Keep line endings stable through `.gitattributes`.
- Treat `global_cnn.pt`, `global_rf.pkl`, and `global_pca.pkl` as verified artifacts.
- Update `RESULTS.md`, `MODEL_CARD.md`, and `ARTIFACTS.md` when changing verified artifacts.
- Keep `scikit-learn==1.7.2` unless PCA/RF artifacts are regenerated and reverified.

## Artifact Changes

Only update model artifacts when you can provide:

- the training/evaluation command,
- dataset split details,
- new metric output,
- updated `RESULTS.md`,
- updated `MODEL_CARD.md`,
- updated `ARTIFACTS.md`,
- updated release or release notes if publishing artifacts.

## Need Ideas?

Useful contribution directions:

- batch inference for folders,
- automatic artifact download from GitHub Releases,
- confusion matrix generation,
- per-class precision/recall,
- small public sample dataset,
- dashboard improvements,
- additional backbones,
- differential privacy experiments,
- secure aggregation experiments.
