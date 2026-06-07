# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
"""
CLIP-based clustering auto-assignment for PtAnalytics.

Clusters unlabeled objects by visual similarity using CLIP embeddings,
then assigns cluster names per class (e.g., person_001, car_001).

Usage:
  from tools.auto_assign_clip import run_auto_assign
  assignments, details = run_auto_assign(extract_dir, manifest, eps=0.5, min_samples=2)
"""
from pathlib import Path
from collections import defaultdict

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


def _compute_pair_embedding(model, preprocess, device, entry, extract_dir):
    """Compute combined embedding from crop+full frame pair (average of both)."""
    embs = []
    for key in ("crop", "full"):
        arcname = entry.get(key)
        if not arcname:
            continue
        img_path = extract_dir / arcname
        if img_path.exists():
            embs.append(_compute_embedding(model, preprocess, device, img_path))
    if not embs:
        return None
    return np.mean(embs, axis=0)


def _cluster_embeddings(embeddings, eps=0.5, min_samples=2):
    """Cluster embeddings using DBSCAN. Returns cluster labels (-1 = noise)."""
    from sklearn.cluster import DBSCAN
    X = np.array(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
    return clustering.labels_


def run_auto_assign(extract_dir, manifest, eps=0.5, min_samples=2):
    extract_dir = Path(extract_dir)
    unlabeled = manifest.get("unlabeled", [])
    if not unlabeled:
        print("No unlabeled images found")
        return {}, []

    print(f"Loading CLIP model...")
    model, preprocess, device = _load_clip()
    print(f"CLIP loaded on {device}")

    # Group by class_name for per-class clustering
    by_class = defaultdict(list)
    for entry in unlabeled:
        by_class[entry.get("class_name", "unknown")].append(entry)

    assignments = {}
    details = []
    total = len(unlabeled)
    assigned_count = 0

    for class_name, entries in sorted(by_class.items()):
        print(f"  Clustering {class_name} ({len(entries)} objects)...")

        # Compute embeddings
        embs = []
        valid_entries = []
        for entry in entries:
            emb = _compute_pair_embedding(model, preprocess, device, entry, extract_dir)
            if emb is not None:
                embs.append(emb)
                valid_entries.append(entry)

        if len(embs) < min_samples:
            print(f"    Too few objects ({len(embs)}), skipping")
            for entry in valid_entries:
                details.append({
                    "object_id": entry["object_id"],
                    "class_name": class_name,
                    "cluster_id": -1,
                    "cluster_name": "",
                    "assigned": False,
                })
            continue

        # Cluster
        labels = _cluster_embeddings(embs, eps=eps, min_samples=min_samples)

        # Assign names per cluster
        cluster_map = {}
        for label in set(labels):
            if label == -1:
                continue
            count = int(np.sum(labels == label))
            if count >= min_samples:
                cluster_id = f"{class_name}_{label + 1:03d}"
                cluster_map[label] = cluster_id

        for entry, label, emb in zip(valid_entries, labels, embs):
            cluster_name = cluster_map.get(label, "")
            assigned = bool(cluster_name)
            details.append({
                "object_id": entry["object_id"],
                "class_name": class_name,
                "cluster_id": int(label),
                "cluster_name": cluster_name,
                "assigned": assigned,
            })
            if assigned:
                assignments[entry["object_id"]] = cluster_name
                assigned_count += 1

        n_clusters = len(cluster_map)
        print(f"    Clusters: {n_clusters}, assigned: {len(cluster_map) * min_samples}+")

    print(f"  Assigned: {assigned_count}/{total}")
    return assignments, details
