import json
import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available torch device for training/inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backbone_training(
    model: torch.nn.Module,
    backbone: str,
    train_backbone: bool,
    trainable_blocks: int,
) -> int:
    """Freeze/unfreeze the backbone and return the count of trainable parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if not train_backbone:
        return 0

    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"} and hasattr(model, "features"):
        blocks = list(model.features.children())
        keep = max(1, min(int(trainable_blocks), len(blocks)))
        for block in blocks[-keep:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_trainable_state_keys(model: torch.nn.Module) -> List[str]:
    """Return state_dict keys for trainable modules, including their buffers."""
    module_prefixes = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            module_prefixes.add(name.rsplit(".", 1)[0] if "." in name else name)

    state_keys: List[str] = []
    for key in model.state_dict().keys():
        key_prefix = key.rsplit(".", 1)[0] if "." in key else key
        if key in module_prefixes or key_prefix in module_prefixes:
            state_keys.append(key)
            continue
        if any(key.startswith(prefix + ".") for prefix in module_prefixes):
            state_keys.append(key)
    return state_keys


def get_parameters_from_model(model: torch.nn.Module, state_keys: Optional[List[str]] = None) -> List:
    """Extract model parameters as a list of numpy arrays."""
    state_dict = model.state_dict()
    keys = state_keys if state_keys is not None else list(state_dict.keys())
    return [state_dict[key].detach().cpu().numpy() for key in keys]


def set_parameters_to_model(
    model: torch.nn.Module,
    parameters: List,
    state_keys: Optional[List[str]] = None,
) -> None:
    """Load a list of numpy arrays into model.state_dict() order."""
    state_dict = model.state_dict()
    keys = state_keys if state_keys is not None else list(state_dict.keys())
    new_state_dict = dict(state_dict)
    for key, parameter in zip(keys, parameters):
        new_state_dict[key] = torch.tensor(parameter)
    model.load_state_dict(new_state_dict, strict=True)


def set_global_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and Torch seeds for reproducible local runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def payload_size_kb(parameters: List[np.ndarray]) -> float:
    """Return payload size in KB for a dense parameter list."""
    total_bytes = sum(int(np.asarray(param).nbytes) for param in parameters)
    return total_bytes / 1024.0


def sparse_payload_size_kb(sparse_params: List[np.ndarray], masks: List[np.ndarray]) -> float:
    """Estimate encoded sparse payload size in KB using values + masks."""
    total_bytes = 0
    for sparse_param, mask in zip(sparse_params, masks):
        sparse_param = np.asarray(sparse_param)
        mask = np.asarray(mask, dtype=np.uint8)
        if np.issubdtype(sparse_param.dtype, np.floating):
            total_bytes += int(mask.sum()) * sparse_param.dtype.itemsize
            total_bytes += mask.nbytes
        else:
            total_bytes += sparse_param.nbytes
    return total_bytes / 1024.0


def sparsify_parameters(params: List[np.ndarray], k: float = 0.01) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Keep top-k% values by magnitude; return (sparse_params, masks)."""
    sparse_params: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    keep_ratio = float(np.clip(k, 0.0, 1.0))

    for param in params:
        array = np.asarray(param)
        if array.size == 0 or not np.issubdtype(array.dtype, np.floating):
            sparse_params.append(array.copy())
            masks.append(np.ones_like(array, dtype=np.uint8))
            continue

        if keep_ratio <= 0.0:
            masks.append(np.zeros_like(array, dtype=np.uint8))
            sparse_params.append(np.zeros_like(array))
            continue

        flat = array.reshape(-1)
        keep_count = max(1, int(np.ceil(flat.size * keep_ratio)))
        if keep_count >= flat.size:
            sparse_params.append(array.copy())
            masks.append(np.ones_like(array, dtype=np.uint8))
            continue

        threshold = np.partition(np.abs(flat), -keep_count)[-keep_count]
        mask = (np.abs(array) >= threshold).astype(np.uint8)
        sparse_params.append((array * mask).astype(array.dtype, copy=False))
        masks.append(mask)

    return sparse_params, masks


def append_optimization_result(result: Dict, out_path: str = "optimization_results.json") -> None:
    """Append one benchmark result to a JSON list on disk."""
    existing: List[Dict] = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []

    existing.append(result)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)


def read_optimization_results(out_path: str = "optimization_results.json") -> List[Dict]:
    """Read optimization benchmark entries if present."""
    if not os.path.exists(out_path):
        return []
    with open(out_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def model_size_kb(model: torch.nn.Module) -> float:
    """Serialize a model state dict to estimate checkpoint size in KB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temp_path = handle.name
    try:
        torch.save(model.state_dict(), temp_path)
        return os.path.getsize(temp_path) / 1024.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
