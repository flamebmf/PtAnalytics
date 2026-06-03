# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
"""
CLIP-based auto-assignment for PtAnalytics.

Processes exported crops from a server, computes embeddings for
reference images (named objects) and unlabeled images, then assigns
names based on cosine similarity.

Usage (called from sync.py):
  from tools.auto_assign_clip import run_auto_assign
  assignments = run_auto_assign(extract_dir, manifest, threshold=0.85)
"""
import json
import sys
from pathlib import Path

import numpy as np


def _load_clip():
    """Lazy-load CLIP model (imported on demand)."""
    import torch
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def _compute_embedding(model, preprocess, device, image_path):
    import torch
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def run_auto_assign(extract_dir, manifest, threshold=0.85):
    extract_dir = Path(extract_dir)

    # Load reference embeddings
    refs = manifest.get("references", {})
    if not refs:
        print("No reference images found in manifest")
        return {}

    print(f"Loading CLIP model...")
    model, preprocess, device = _load_clip()
    print(f"CLIP loaded on {device}")

    # Compute reference embeddings (average per name)
    ref_embeddings = {}
    for name, arcnames in refs.items():
        embs = []
        for arcname in arcnames:
            img_path = extract_dir / arcname
            if not img_path.exists():
                continue
            emb = _compute_embedding(model, preprocess, device, img_path)
            embs.append(emb)
        if embs:
            ref_embeddings[name] = np.mean(embs, axis=0)

    if not ref_embeddings:
        print("No reference embeddings computed")
        return {}

    ref_names = list(ref_embeddings.keys())
    ref_matrix = np.array([ref_embeddings[n] for n in ref_names])

    # Process unlabeled
    unlabeled = manifest.get("unlabeled", [])
    if not unlabeled:
        print("No unlabeled images found")
        return {}

    assignments = {}
    ref_idx = 0
    total = len(unlabeled)
    for entry in unlabeled:
        ref_idx += 1
        if ref_idx % 10 == 0:
            print(f"  processed {ref_idx}/{total}")
        img_path = extract_dir / entry["arcname"]
        if not img_path.exists():
            continue
        emb = _compute_embedding(model, preprocess, device, img_path)
        sims = ref_matrix @ emb  # cosine similarity (normalized)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            assignments[entry["object_id"]] = ref_names[best_idx]

    print(f"  processed {total}/{total} — {len(assignments)} assigned")
    return assignments
