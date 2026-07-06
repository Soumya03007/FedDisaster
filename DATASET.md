# Dataset Preparation

FedDisaster expects image data in `torchvision.datasets.ImageFolder` format. Each class is a folder, and each folder contains images for that class.

## Required Layout

```text
data/
  client_1/
    train/
      Damaged_Infrastructure/
      Fire_Disaster/
      Human_Damage/
      Land_Disaster/
      Non_Damage/
      Water_Disaster/
    test/
      Damaged_Infrastructure/
      Fire_Disaster/
      Human_Damage/
      Land_Disaster/
      Non_Damage/
      Water_Disaster/
  client_2/
    train/
    test/
  client_3/
    train/
    test/
  global_test/
    Damaged_Infrastructure/
    Fire_Disaster/
    Human_Damage/
    Land_Disaster/
    Non_Damage/
    Water_Disaster/
```

Every client and `data/global_test` must use the same class folder names. This keeps class indices consistent across federated clients, global evaluation, and inference.

## Verified Classes

The verified artifact release uses these 6 classes:

```text
Damaged_Infrastructure
Fire_Disaster
Human_Damage
Land_Disaster
Non_Damage
Water_Disaster
```

## Creating A Small Local Sample

For quick local smoke tests, create a tiny subset with a few images per class. Keep the folder names identical:

```text
data_sample/
  global_test/
    Damaged_Infrastructure/
    Fire_Disaster/
    Human_Damage/
    Land_Disaster/
    Non_Damage/
    Water_Disaster/
```

Then run single-image inference against any image:

```bash
python predict.py --image data_sample/global_test/Water_Disaster/example.jpg --artifacts release
```

If you want to evaluate a full global test folder, use:

```bash
python scripts/evaluate_best_artifacts.py --backbone_path global_cnn.pt --rf_path global_rf.pkl --pca_path global_pca.pkl --batch_size 64
```

## Preparing Client Splits

If your raw data is already organized by class, copy or split it into:

```text
data/client_<id>/train/<class_name>/
data/client_<id>/test/<class_name>/
data/global_test/<class_name>/
```

The helper script can distribute multi-source data:

```bash
python data/setup_multiclass_dataset.py \
  --disaster_sources flood=data/_organized fire=path/to/fire landslide=path/to/landslide \
  --target_root data \
  --num_clients 3 \
  --force
```

## Public Repository Guidance

Avoid committing large or private raw datasets. Prefer one of:

- publish dataset instructions and links,
- include only a tiny non-sensitive sample dataset,
- keep full datasets outside Git and document the expected layout.

The verified model artifacts are distributed through GitHub Releases:

```text
https://github.com/Soumya03007/FedDisaster/releases/tag/v0.1.0-efficientnet-rf
```
