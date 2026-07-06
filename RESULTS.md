# Results

This file records the verified artifact result for the current repository state.

## Verified Best Artifact

- Backbone: EfficientNet-B0
- Federated method: Flower FedAvg over shared backbone parameters
- Local client classifier: private linear local head
- Global classifier: PCA + RandomForest trained on extracted EfficientNet features
- Number of classes: 6
- Global test samples: 1,353
- Verified global RF accuracy: 94.0872%
- Best recorded client-side accuracy: 95.2243%
- Source run: `runs/metrics_20260423_131506.json`
- Preserved artifacts:
  - `best_artifacts/global_backbone_best.pt`
  - `best_artifacts/global_pca_best.pkl`
  - `best_artifacts/global_rf_best.pkl`

## Verification Command

Run from the repository root after installing dependencies:

```bash
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

Expected output:

```text
classes=6
samples=1353
accuracy=0.940872
```

## Historical Runs

The strongest saved run is `runs/metrics_20260423_131506.json`.

Its RF accuracy trajectory was:

```text
0.8921, 0.8921, 0.8921, 0.8921, 0.8921,
0.8921, 0.8921, 0.8921, 0.8921, 0.9409
```

Its weighted client-side accuracy trajectory was:

```text
0.9000, 0.9184, 0.9233, 0.9404, 0.9404,
0.9449, 0.9449, 0.9449, 0.9498, 0.9522
```

## Notes

- The reported accuracy is for the saved EfficientNet-B0 backbone and saved PCA+RandomForest artifacts.
- `simplecnn` remains in the project for compatibility and smoke testing, but it is not the target model for the reported result.
- Global RF evaluation uses a held-out `data/global_test` folder with the same class mapping as the client datasets.
- The global RandomForest is centralized over extracted features. Raw client images are not exchanged during federated training.
