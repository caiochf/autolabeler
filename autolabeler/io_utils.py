from __future__ import annotations
import shutil
from pathlib import Path
from typing import List, Dict


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_support_images(support_dir: str) -> Dict[str, List[Path]]:
    """
    Expected structure:
      support_dir/
        classA/*.png
        classB/*.png

    Returns:
      dict: class_name -> list of image Paths
    """
    base = Path(support_dir)
    if not base.exists():
        raise FileNotFoundError(f"Support directory does not exist: {support_dir}")

    out: Dict[str, List[Path]] = {}
    for sub in sorted([p for p in base.iterdir() if p.is_dir()]):
        cls = sub.name
        imgs = [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if imgs:
            out[cls] = sorted(imgs)

    if not out:
        raise ValueError("No class subfolders with images found inside support_dir.")
    return out


def list_query_images(query_dir: str, recursive: bool = True) -> List[Path]:
    """
    Lists images inside query_dir.
    If recursive=True, scans subfolders recursively.
    If recursive=False, scans only the top-level directory.
    """
    base = Path(query_dir)
    if not base.exists():
        raise FileNotFoundError(f"Query directory does not exist: {query_dir}")

    it = base.rglob("*") if recursive else base.iterdir()
    imgs = [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS]

    if not imgs:
        mode = "recursive" if recursive else "flat"
        raise ValueError(f"No images found inside query_dir (mode={mode}).")
    return sorted(imgs)


def ensure_class_folders(output_dir: str, class_names: List[str]) -> None:
    """
    Creates output_dir/class_name for each class name.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    for c in class_names:
        (base / c).mkdir(parents=True, exist_ok=True)

def ensure_ok_not_sure_folders(output_dir: str, class_names: List[str]) -> None:
    """
    Creates:
      output_dir/ok/<class_name>/
      output_dir/not_sure/<class_name>/
    """
    base = Path(output_dir)
    for branch in ["ok", "not_sure"]:
        (base / branch).mkdir(parents=True, exist_ok=True)
        for c in class_names:
            (base / branch / c).mkdir(parents=True, exist_ok=True)


def copy_to_class_folder(output_dir: str, class_name: str, img_path: Path) -> Path:
    """
    Copies img_path into output_dir/class_name/
    """
    dest = Path(output_dir) / class_name / img_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest)
    return dest

def copy_to_branch_class_folder(output_dir: str, branch: str, class_name: str, img_path: Path) -> Path:
    """
    Copies img_path into:
      output_dir/<branch>/<class_name>/
    """
    dest = Path(output_dir) / branch / class_name / img_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest)
    return dest


