# Model Card

## Model Summary

FedDisaster is a federated multi-class disaster image classification pipeline. Clients train using local image folders, Flower coordinates federated averaging over shared EfficientNet-B0 backbone parameters, and the server evaluates global performance using PCA + RandomForest over extracted backbone features.

## Intended Use

This project is intended for research, prototyping, and educational work around:

- Federated image classification
- Disaster-scene image categorization
- Client-local training with central feature-based evaluation
- Reproducible ML artifact workflows

It is not intended for direct emergency-response deployment without additional validation, monitoring, and domain review.

## Architecture

- Backbone: EfficientNet-B0
- Federated strategy: Flower FedAvg
- Client-local classifier: private linear classification head
- Global evaluator: PCA + RandomForest
- Input format: `torchvision.datasets.ImageFolder`
- Data layout: per-client train/test folders plus a held-out `data/global_test`

## Classes

The verified artifact was evaluated over 6 classes:

```text
Damaged_Infrastructure
Fire_Disaster
Human_Damage
Land_Disaster
Non_Damage
Water_Disaster
```

## Verified Performance

- Global test samples: 1,353
- Verified global RF accuracy: 94.0872%
- Best recorded weighted client-side accuracy: 95.2243%
- Source run: `runs/metrics_20260423_131506.json`

Verification command:

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

Expected output:

```text
classes=6
samples=1353
accuracy=0.940872
```

## Limitations

- Accuracy depends on the exact dataset split and class-folder consistency.
- The global RandomForest is centralized over extracted features; it is not itself federated.
- Saved `.pkl` artifacts should be loaded only from trusted sources.
- Disaster imagery can be geographically and visually biased depending on dataset composition.
- This model should not be used as a sole source of truth for safety-critical disaster response decisions.

## Reproducibility

Use the pinned dependencies in `requirements.txt`. The saved PCA and RandomForest artifacts were generated with `scikit-learn==1.7.2`.

For artifact details and release packaging, see `ARTIFACTS.md`.
