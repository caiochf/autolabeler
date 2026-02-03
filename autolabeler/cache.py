from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch


@dataclass
class PrototypeCache:
    class_names: List[str]
    prototypes: Dict[str, torch.Tensor]  # backbone_key -> (C, D)
    image_size: int
    backbones: List[str]

    def to_dict(self) -> dict:
        return {
            "class_names": self.class_names,
            "prototypes": self.prototypes,
            "image_size": self.image_size,
            "backbones": self.backbones,
        }

    @staticmethod
    def from_dict(d: dict) -> "PrototypeCache":
        return PrototypeCache(
            class_names=d["class_names"],
            prototypes=d["prototypes"],
            image_size=int(d["image_size"]),
            backbones=list(d["backbones"]),
        )


def save_prototypes(path: str, cache: PrototypeCache) -> None:
    torch.save(cache.to_dict(), path)


def load_prototypes(path: str) -> PrototypeCache:
    d = torch.load(path, map_location="cpu")
    return PrototypeCache.from_dict(d)

