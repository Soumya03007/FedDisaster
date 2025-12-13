import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Federated CNN Feature Extractor (NO classifier head).
    Used only to extract features for the centralized Random Forest.
    Feature dimension is computed dynamically.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # ✅ Convolutional Backbone
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=6, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=6, padding=1)

        # ✅ NEW: Third convolutional layer for stronger features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # ✅ Dropout for realistic generalization
        self.dropout = nn.Dropout(0.3)

        # ✅ SAFE: Auto-compute feature dimension
        self.feature_dim = self._compute_feature_dim()

    def _compute_feature_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)  # Simulated input image
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.dropout(x)

            x = self.pool(F.relu(self.conv2(x)))
            x = self.dropout(x)

            x = self.pool(F.relu(self.conv3(x)))
            x = self.dropout(x)

            return x.view(1, -1).shape[1]  # ✅ Dynamic feature size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        return x   # ✅ Features only


class EfficientNetB0Extractor(nn.Module):
    """EfficientNet-B0 feature extractor (no classifier head).

    Notes:
    - For best results, resize inputs to ~224x224 and use ImageNet normalization.
    - feature_dim is 1280 for EfficientNet-B0.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Import lazily to avoid hard dependency issues during tooling/linters
        try:
            from torchvision.models import efficientnet_b0
        except Exception as e:
            raise ImportError("torchvision is required for EfficientNetB0Extractor") from e

        # Handle torchvision version differences: `weights=` (new) vs `pretrained=` (old)
        weights = None
        if pretrained:
            try:
                from torchvision.models import EfficientNet_B0_Weights

                weights = EfficientNet_B0_Weights.DEFAULT
            except Exception:
                weights = None

        try:
            model = efficientnet_b0(weights=weights)
        except TypeError:
            # Older torchvision
            model = efficientnet_b0(pretrained=pretrained)

        self.features = model.features
        self.avgpool = model.avgpool  # AdaptiveAvgPool2d(1)

        # EfficientNet-B0 classifier is typically: Dropout -> Linear(1280, 1000)
        try:
            self.feature_dim = model.classifier[1].in_features
        except Exception:
            self.feature_dim = 1280

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x


# Optional: Local client training head (clients only, NOT shared)
class LocalHead(nn.Module):
    """
    Used only on clients for supervised training.
    NEVER sent to the server.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
