# Artifacts

This project uses saved model artifacts to make the best recorded result reproducible without rerunning federated training.

## Recommended Distribution

For public collaboration, publish the verified artifacts as GitHub Release assets instead of relying only on repository-tracked binary files.

Recommended release tag:

```text
v0.1.0-efficientnet-rf
```

Recommended release title:

```text
FedDisaster EfficientNet-B0 + RandomForest Artifacts
```

Published release:

```text
https://github.com/Soumya03007/FedDisaster/releases/tag/v0.1.0-efficientnet-rf
```

## Verified Artifact Set

The current verified top-level files are:

| File | Purpose |
| --- | --- |
| `global_cnn.pt` | EfficientNet-B0 backbone weights used for feature extraction |
| `global_pca.pkl` | PCA transformer fitted on extracted training features |
| `global_rf.pkl` | RandomForest classifier fitted on PCA-transformed features |

The preserved best artifact copies are:

| File | Purpose |
| --- | --- |
| `best_artifacts/global_backbone_best.pt` | Best saved EfficientNet-B0 backbone |
| `best_artifacts/global_pca_best.pkl` | Best saved PCA transformer |
| `best_artifacts/global_rf_best.pkl` | Best saved RandomForest classifier |
| `best_artifacts/best_artifacts.json` | Metadata pointing to the best source run |

## Verification

Run:

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

Expected output:

```text
classes=6
samples=1353
accuracy=0.940872
```

## Release Bundle

Recommended archive name:

```text
feddisaster-efficientnet-rf-artifacts-v0.1.0.tar.gz
```

Recommended contents:

```text
global_cnn.pt
global_pca.pkl
global_rf.pkl
best_artifacts/global_backbone_best.pt
best_artifacts/global_pca_best.pkl
best_artifacts/global_rf_best.pkl
best_artifacts/best_artifacts.json
RESULTS.md
MODEL_CARD.md
ARTIFACTS.md
```

## Notes For Maintainers

- Update artifacts only when `RESULTS.md` is updated with the new verified metric.
- Keep `scikit-learn==1.7.2` unless RF/PCA artifacts are regenerated.
- If artifacts move fully to GitHub Releases later, leave download instructions in this file and avoid committing new large binaries to normal Git history.
