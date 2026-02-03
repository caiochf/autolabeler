from __future__ import annotations

import argparse
from pathlib import Path
import torch

from .io_utils import list_support_images, list_query_images, ensure_ok_not_sure_folders
from .models import Backbone
from .embedder import embed_paths
from .prototypes import compute_class_prototypes
from .classify import cosine_scores, top1, decide_final_label, materialize_output, write_csv_generic
from .cache import PrototypeCache, save_prototypes, load_prototypes


def main():
    parser = argparse.ArgumentParser("autolabeler")

    # Core paths
    parser.add_argument("--support_dir", type=str, required=True)
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default="results.csv")

    # Prototype cache
    parser.add_argument("--prototypes_path", type=str, default="prototypes_cache.pt")
    parser.add_argument("--recompute_prototypes", action="store_true")

    # Embedding params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)

    # Decision params
    parser.add_argument("--threshold_x", type=float, default=0.35)
    parser.add_argument("--min_votes", type=int, default=2)
    parser.add_argument("--fallback", type=str, choices=["best", "unlabeled"], default="best")

    # Query scanning
    parser.add_argument("--query_recursive", action=argparse.BooleanOptionalAction, default=True)

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # ---------------------------------------------------------------------
    # Model registry: (cache_key, backbone_name)
    # Add/remove models here (keys become CSV prefixes and cache keys).
    # ---------------------------------------------------------------------
    MODEL_SPECS = [
        ("eff", "efficientnetv2_s"),
        ("resnet", "resnet50"),
        ("vgg", "vgg16"),
        ("densenet", "densenet121"),
        ("convnext", "convnext_tiny"),
    ]

    # Step 1: read support directory (class folders)
    support = list_support_images(args.support_dir)
    class_names = sorted(support.keys())

    # Prepare output folders (ok / not_sure branches)
    ensure_ok_not_sure_folders(args.output_dir, class_names)

    # Step 2: build or load prototypes
    cache_path = Path(args.prototypes_path)
    use_cache = cache_path.exists() and not args.recompute_prototypes

    if use_cache:
        cache = load_prototypes(str(cache_path))

        if cache.image_size != args.image_size:
            raise ValueError(
                f"Cache image_size={cache.image_size} does not match args.image_size={args.image_size}. "
                "Recompute prototypes with --recompute_prototypes."
            )

        # Ensure cache includes all requested keys
        missing = [k for k, _ in MODEL_SPECS if k not in cache.prototypes]
        if missing:
            raise ValueError(
                f"Prototype cache is missing keys: {missing}. "
                "Recompute prototypes with --recompute_prototypes."
            )

        class_names = cache.class_names
        proto_by_key = cache.prototypes
        print(f"[Cache] Loaded prototypes from: {cache_path}")

    else:
        proto_by_key = {}
        class_names_ref = None

        for key, backbone_name in MODEL_SPECS:
            backbone = Backbone(backbone_name, device=device)

            def embed_fn(paths, _backbone=backbone, _name=backbone_name):
                return embed_paths(
                    _backbone,
                    paths=paths,
                    batch_size=args.batch_size,
                    device=device,
                    image_size=args.image_size,
                    desc=f"Embedding support ({_name})"
                )

            classes, protos = compute_class_prototypes(support, embed_fn)

            if class_names_ref is None:
                class_names_ref = classes
            elif classes != class_names_ref:
                raise RuntimeError(
                    "Class order mismatch across backbones. "
                    "Check support folder structure and ensure all classes are consistent."
                )

            proto_by_key[key] = protos

        class_names = class_names_ref

        cache = PrototypeCache(
            class_names=class_names,
            prototypes=proto_by_key,
            image_size=args.image_size,
            backbones=[name for _, name in MODEL_SPECS],
        )
        save_prototypes(str(cache_path), cache)
        print(f"[Cache] Saved prototypes to: {cache_path}")

    # Step 3: list query images
    query_paths = list_query_images(args.query_dir, recursive=args.query_recursive)

    # Step 3: embed queries per backbone
    query_emb_by_key = {}
        
    for key, backbone_name in MODEL_SPECS:
        backbone = Backbone(backbone_name, device=device)
        query_emb_by_key[key] = embed_paths(
            backbone,
            query_paths,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
            desc=f"Embedding queries ({backbone_name})"
        )

    # Step 4: cosine similarity and top-1 prediction per model
    pred_by_key = {}
    for key, _backbone_name in MODEL_SPECS:
        scores = cosine_scores(query_emb_by_key[key], proto_by_key[key])  # (N, C)
        pred_by_key[key] = top1(scores, class_names)

    # Step 6: voting rule (min_votes out of N models)
    final = []
    for i in range(len(query_paths)):
        results = [(key, pred_by_key[key][i]) for key, _ in MODEL_SPECS]
        final.append(
            decide_final_label(
                results,
                threshold_x=args.threshold_x,
                min_votes=args.min_votes,
                fallback=args.fallback
            )
        )

    # Step 5: write CSV (dynamic number of models)
    write_csv_generic(args.csv_path, query_paths, pred_by_key, final)

    # Step 7: copy images into ok/ vs not_sure/ by reason
    copied = materialize_output(
        args.output_dir,
        query_paths,
        final,
        allowed_classes=class_names
    )

    print(f"Done. CSV saved at: {args.csv_path} | Copied images: {copied}/{len(query_paths)}")


if __name__ == "__main__":
    main()

