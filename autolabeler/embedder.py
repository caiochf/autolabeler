from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def build_transform(image_size: int = 224) -> transforms.Compose:
    """
    Standard ImageNet preprocessing for full-image embedding.

    Note: This resizes the full image to (image_size, image_size).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image_rgb(path: Path) -> Image.Image:
    """
    Loads an image file and converts it to RGB (3-channel).
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


@torch.inference_mode()
def embed_paths(
    model,
    paths: List[Path],
    batch_size: int,
    device: torch.device,
    image_size: int = 224,
    desc: str = "Embedding",
) -> torch.Tensor:
    """
    Embeds a list of image paths using full-image resizing (no tiling).

    Returns:
      embeddings: Tensor (N, D) on CPU.
    """
    tfm = build_transform(image_size=image_size)
    model.eval()

    chunks = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, unit="batch"):
        batch_paths = paths[i:i + batch_size]
        batch_imgs = [tfm(load_image_rgb(p)) for p in batch_paths]
        x = torch.stack(batch_imgs, dim=0).to(device)  # (B, 3, H, W)
        e = model(x)  # (B, D)
        if e.ndim > 2:
            e = torch.flatten(e, start_dim=1)
        chunks.append(e.detach().cpu())

    return torch.cat(chunks, dim=0)

