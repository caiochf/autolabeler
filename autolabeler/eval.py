from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_ground_truth_map(gabarito_dir: str) -> Dict[str, str]:
    """
    Builds a map: filename -> true_class from a folder structure like:
      gabarito/class_name/image.png

    NOTE: This assumes filenames are unique across all classes.
    If you have duplicates, switch to relative-path matching.
    """
    base = Path(gabarito_dir)
    if not base.exists():
        raise FileNotFoundError(f"gabarito_dir does not exist: {gabarito_dir}")

    gt: Dict[str, str] = {}
    for class_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        cls = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                key = p.name  # filename only
                if key in gt:
                    raise ValueError(
                        f"Duplicate filename found in gabarito: {key}. "
                        "Use relative-path matching instead."
                    )
                gt[key] = cls

    if not gt:
        raise ValueError("No images found inside gabarito_dir.")
    return gt


def confusion_and_report(y_true: List[str], y_pred: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Returns:
      - confusion matrix (DataFrame)
      - per-class report (precision/recall/f1/support)
      - accuracy (float)

    Implemented with pandas only (no sklearn dependency).
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    labels = sorted(set(y_true) | set(y_pred))

    cm = pd.crosstab(
        df["y_true"],
        df["y_pred"],
        rownames=["true"],
        colnames=["pred"],
        dropna=False
    ).reindex(index=labels, columns=labels, fill_value=0)

    report_rows = []
    for cls in labels:
        tp = int(((df["y_true"] == cls) & (df["y_pred"] == cls)).sum())
        fp = int(((df["y_true"] != cls) & (df["y_pred"] == cls)).sum())
        fn = int(((df["y_true"] == cls) & (df["y_pred"] != cls)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int((df["y_true"] == cls).sum())

        report_rows.append({
            "class": cls,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })

    report = pd.DataFrame(report_rows).sort_values("class").reset_index(drop=True)
    accuracy = float((df["y_true"] == df["y_pred"]).mean())

    return cm, report, accuracy


def main():
    ap = argparse.ArgumentParser("autolabeler-eval")
    ap.add_argument("--gabarito_dir", required=True, type=str)
    ap.add_argument("--results_csv", required=True, type=str)
    ap.add_argument("--pred_col", default="final_pred", type=str)

    ap.add_argument(
        "--ignore_pred",
        default="",
        type=str,
        help='If set, ignore rows where pred_col == this value (e.g. "_unlabeled").'
    )

    ap.add_argument(
        "--only_ok",
        action="store_true",
        help="Evaluate only rows whose final_reason starts with 'vote' (i.e., ok branch)."
    )

    ap.add_argument(
        "--save_misclassified_csv",
        default="misclassified.csv",
        type=str
    )

    args = ap.parse_args()

    gt = build_ground_truth_map(args.gabarito_dir)

    df = pd.read_csv(args.results_csv)
    if args.pred_col not in df.columns:
        raise ValueError(f"Column '{args.pred_col}' not found in {args.results_csv}")

    if args.only_ok and "final_reason" not in df.columns:
        raise ValueError("final_reason column not found in results CSV, but --only_ok was requested.")

    # Extract filename from image_path
    df["filename"] = df["image_path"].apply(lambda s: Path(s).name)

    # Attach ground-truth labels
    df["true_class"] = df["filename"].map(gt)

    # Keep only rows that exist in gabarito
    df_eval_all = df.dropna(subset=["true_class"]).copy()

    # Optional: ignore a specific predicted label
    if args.ignore_pred:
        df_eval_all = df_eval_all[df_eval_all[args.pred_col] != args.ignore_pred].copy()

    if len(df_eval_all) == 0:
        raise ValueError("No overlap between results.csv and gabarito images (after filtering).")

    # Optional: evaluate only "ok" (vote-based) predictions
    if args.only_ok:
        df_eval = df_eval_all[df_eval_all["final_reason"].astype(str).str.startswith("vote")].copy()
    else:
        df_eval = df_eval_all

    if len(df_eval) == 0:
        raise ValueError("After filtering, there are no samples left to evaluate.")

    y_true = df_eval["true_class"].tolist()
    y_pred = df_eval[args.pred_col].tolist()

    cm, report, acc = confusion_and_report(y_true, y_pred)

    # Coverage reporting (useful when only_ok=True)
    coverage = len(df_eval) / len(df_eval_all)

    print(f"Total evaluable samples (with GT): {len(df_eval_all)}")
    print(f"Evaluated samples (after filters): {len(df_eval)}")
    print(f"Coverage: {coverage:.4f}")
    print(f"Accuracy: {acc:.4f}\n")

    print("Per-class report:")
    print(report.to_string(index=False))
    print("\nConfusion matrix:")
    print(cm.to_string())

    # Save misclassified list
    mis = df_eval[df_eval["true_class"] != df_eval[args.pred_col]].copy()
    mis.to_csv(args.save_misclassified_csv, index=False)
    print(f"\nSaved misclassified samples to: {args.save_misclassified_csv}")


if __name__ == "__main__":
    main()

