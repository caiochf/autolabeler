from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import torch


def compute_class_prototypes(
    class_to_paths: Dict[str, List[Path]],
    embed_fn
) -> Tuple[List[str], torch.Tensor]:
    """
    Computes one prototype (mean embedding) per class.

    embed_fn(paths) must return Tensor (N, D)

    Returns:
      class_names: list of classes (sorted)
      prototypes: Tensor (C, D)
    """
    class_names = sorted(class_to_paths.keys())
    protos = []

    for cls in class_names:
        embs = embed_fn(class_to_paths[cls])  # (N,D)
        proto = embs.mean(dim=0)              # (D,)
        protos.append(proto)

    return class_names, torch.stack(protos, dim=0)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalizes vectors along the last dimension.
    Needed for cosine similarity using dot product.
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)

