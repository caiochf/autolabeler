from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
import csv
import torch

from .prototypes import l2_normalize
from .io_utils import copy_to_class_folder, copy_to_branch_class_folder

@dataclass
class ModelResult:
    pred_class: str
    score: float


def cosine_scores(query_embs: torch.Tensor, proto_embs: torch.Tensor) -> torch.Tensor:
    """
    query_embs: (N, D)
    proto_embs: (C, D)
    returns: (N, C) cosine similarities
    """
    q = l2_normalize(query_embs)
    p = l2_normalize(proto_embs)
    return q @ p.T


def top1(scores: torch.Tensor, class_names: List[str]) -> List[ModelResult]:
    """
    Extracts top-1 class and score per query.
    scores: (N, C)
    """
    vals, idx = torch.max(scores, dim=1)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        out.append(ModelResult(pred_class=class_names[i], score=float(v)))
    return out


def decide_final_label(
    results: List[Tuple[str, ModelResult]],
    threshold_x: float,
    min_votes: int = 2,
    fallback: str = "best",
    unlabeled_name: str = "_unlabeled"
) -> Tuple[str, float, str]:
    """
    Returns:
      (final_class, final_score, reason)

    reason will start with:
      - "vote" when voting succeeded  -> OK branch
      - "fallback" when fallback used -> NOT_SURE branch
    """
    eligible = [(name, r) for name, r in results if r.score >= threshold_x]

    counts: Dict[str, List[Tuple[str, float]]] = {}
    for name, r in eligible:
        counts.setdefault(r.pred_class, []).append((name, r.score))

    for cls, lst in counts.items():
        if len(lst) >= min_votes:
            final_score = float(sum(s for _, s in lst) / len(lst))
            reason = f"vote(min_votes={min_votes};models={','.join(n for n,_ in lst)})"
            return cls, final_score, reason

    if fallback == "unlabeled":
        return unlabeled_name, -1.0, f"fallback_unlabeled(min_votes={min_votes})"

    best_name, best_res = max(results, key=lambda kv: kv[1].score)
    return best_res.pred_class, best_res.score, f"fallback_best({best_name})"

def write_csv_generic(
    csv_path: str,
    query_paths: List[Path],
    pred_by_key: Dict[str, List[ModelResult]],
    final: List[Tuple[str, float, str]]
) -> None:
    """
    Writes a CSV with dynamic number of models.
    Columns:
      image_path,
      <key>_pred, <key>_score for each model key,
      final_pred, final_score, final_reason
    """
    model_keys = list(pred_by_key.keys())

    header = ["image_path"]
    for k in model_keys:
        header += [f"{k}_pred", f"{k}_score"]
    header += ["final_pred", "final_score", "final_reason"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for idx, p in enumerate(query_paths):
            row = [str(p)]
            for k in model_keys:
                r = pred_by_key[k][idx]
                row += [r.pred_class, f"{r.score:.6f}"]
            final_cls, final_score, reason = final[idx]
            row += [final_cls, f"{final_score:.6f}", reason]
            w.writerow(row)


def materialize_output(
    output_dir: str,
    query_paths: List[Path],
    final: List[Tuple[str, float, str]],
    allowed_classes: List[str],
) -> int:
    """
    Copies images into:
      output_dir/ok/<class>         if voting succeeded
      output_dir/not_sure/<class>   if fallback was triggered

    We decide branch by reading the 'reason' string:
      - startswith("vote")     -> ok
      - startswith("fallback") -> not_sure
    """
    n = 0
    for p, (cls, score, reason) in zip(query_paths, final):
        if cls not in allowed_classes:
            # If you still want to allow _unlabeled, you can add it to allowed_classes.
            continue

        branch = "ok" if reason.startswith("vote") else "not_sure"
        copy_to_branch_class_folder(output_dir, branch, cls, p)
        n += 1
    return n

