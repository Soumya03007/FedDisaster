# Contributing

Thanks for taking an interest in FedDisaster. This project is intended to be reproducible, useful, and straightforward to inspect.

## Development Setup

Create and activate a virtual environment, then install dependencies:

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

## Verify The Saved Result

After setup, run:

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

Expected result:

```text
accuracy=0.940872
```

## Before Opening A Pull Request

Please run these checks from the repository root:

```bash
python -m compileall client.py server.py dataset_loader.py models.py simple_demo.py utils.py scripts
python -m py_compile predict.py
python -m json.tool metrics.json
python -m json.tool best_metrics.json
```

If your change affects training or evaluation, include:

- The command you ran
- The dataset split used
- The final metric output
- Any changed artifacts or run files

## Repository Hygiene

- Do not commit local virtual environments such as `.venv/` or `.venv-linux/`.
- Do not use `git add .` when generated artifacts or run outputs changed unexpectedly.
- Keep line endings stable through `.gitattributes`.
- Treat `global_cnn.pt`, `global_rf.pkl`, and `global_pca.pkl` as verified artifacts; only update them with a matching result note in `RESULTS.md`.

## Pull Request Scope

Good PRs are focused. Prefer one of:

- Model/training change
- Dataset preparation improvement
- Documentation update
- Experiment artifact update
- Dashboard/metrics update

Avoid mixing large model artifact changes with unrelated code formatting.
