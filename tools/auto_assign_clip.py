# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
"""
CLIP-based clustering auto-assignment for PtAnalytics.

Clusters unlabeled objects by visual similarity using CLIP embeddings,
then assigns names based on existing named references or cluster IDs.

First, unlabeled objects are compared against existing named references
(same class only) via CLIP cosine similarity.  If similarity >= sim_threshold,
the reference name is reused.  Remaining objects are clustered by DBSCAN
and named with auto-increment IDs (e.g., person_001, car_001).

Usage:
  from tools.auto_assign_clip import run_auto_assign
  assignments, details = run_auto_assign(
      extract_dir, manifest, eps=0.5, min_samples=2, sim_threshold=0.85)
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


def _compute_reference_centroids(model, preprocess, device, manifest, extract_dir):
    """Build a dict: class_name -> [(name, centroid_embedding), ...]"""
    references = manifest.get("references", {})
    centroids = defaultdict(list)
    for name, entries in references.items():
        embs = []
        for entry in entries:
            emb = _compute_pair_embedding(model, preprocess, device, entry, extract_dir)
            if emb is not None:
                embs.append(emb)
        if not embs:
            continue
        centroid = np.mean(embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        class_name = entries[0].get("class_name", "unknown") if entries else "unknown"
        centroids[class_name].append((name, centroid))
    return dict(centroids)


def _match_by_reference(entry_emb, ref_centroids, class_name, sim_threshold):
    """Return (name, similarity) of best-matching reference, or (None, 0).

    Tries same-class references first; falls back to all references in case
    the reference lacks class_name (old export format) or the YOLO class
    label was wrong.
    """
    candidates = ref_centroids.get(class_name, [])
    # Fall back to all references if no match in same class
    if not candidates:
        all_refs = []
        for refs in ref_centroids.values():
            all_refs.extend(refs)
        candidates = all_refs
    if not candidates:
        return None, 0.0
    best_name = None
    best_sim = 0.0
    for name, centroid in candidates:
        sim = float(np.dot(entry_emb, centroid))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim >= sim_threshold:
        return best_name, best_sim
    return None, best_sim


def run_auto_assign(extract_dir, manifest, eps=0.5, min_samples=2, sim_threshold=0.85):
    extract_dir = Path(extract_dir)
    unlabeled = manifest.get("unlabeled", [])
    if not unlabeled:
        print("No unlabeled images found")
        return {}, []

    print(f"Loading CLIP model...")
    model, preprocess, device = _load_clip()
    print(f"CLIP loaded on {device}")

    # ---- Step 1: build reference centroids from named objects ----
    print("Computing reference embeddings...")
    ref_centroids = _compute_reference_centroids(
        model, preprocess, device, manifest, extract_dir)
    ref_count = sum(len(v) for v in ref_centroids.values())
    print(f"  References loaded: {ref_count} names across {len(ref_centroids)} classes")

    # ---- Step 2: try to match unlabeled against references ----
    print(f"Matching against references (threshold={sim_threshold})...")
    match_assignments = {}
    match_details = []
    leftover_entries = []
    for entry in unlabeled:
        emb = _compute_pair_embedding(model, preprocess, device, entry, extract_dir)
        class_name = entry.get("class_name", "unknown")
        if emb is None:
            match_details.append({
                "object_id": entry["object_id"],
                "class_name": class_name,
                "cluster_id": -1,
                "cluster_name": "",
                "assigned": False,
                "similarity": 0,
                "match_method": "no_embedding",
            })
            continue
        name, sim = _match_by_reference(emb, ref_centroids, class_name, sim_threshold)
        if name:
            match_assignments[entry["object_id"]] = name
            match_details.append({
                "object_id": entry["object_id"],
                "class_name": class_name,
                "cluster_id": -1,
                "cluster_name": name,
                "assigned": True,
                "similarity": round(sim, 4),
                "match_method": "reference",
            })
        else:
            leftover_entries.append((entry, emb))

    print(f"  Matched by reference: {len(match_assignments)}")

    # ---- Step 3: cluster remaining objects ----
    assignments = {}
    details = []
    assigned_count = 0

    by_class = defaultdict(list)
    for entry, emb in leftover_entries:
        by_class[entry.get("class_name", "unknown")].append((entry, emb))

    # Find highest existing cluster number per class for naming continuity
    max_cluster_per_class = {}
    for class_name in by_class:
        max_cluster_per_class[class_name] = 0

    for class_name, items in sorted(by_class.items()):
        entries, embs = zip(*items) if items else ([], [])
        print(f"  Clustering {class_name} ({len(entries)} remaining)...")

        if len(embs) < min_samples:
            print(f"    Too few objects ({len(embs)}), skipping")
            for entry in entries:
                details.append({
                    "object_id": entry["object_id"],
                    "class_name": class_name,
                    "cluster_id": -1,
                    "cluster_name": "",
                    "assigned": False,
                    "similarity": 0,
                    "match_method": "cluster",
                })
            continue

        labels = _cluster_embeddings(embs, eps=eps, min_samples=min_samples)
        cluster_map = {}
        next_id = max_cluster_per_class.get(class_name, 0)
        for label in set(labels):
            if label == -1:
                continue
            count = int(np.sum(labels == label))
            if count >= min_samples:
                next_id += 1
                cluster_map[label] = f"{class_name}_{next_id:03d}"
        max_cluster_per_class[class_name] = next_id

        for entry, label, emb in zip(entries, labels, embs):
            cluster_name = cluster_map.get(label, "")
            assigned = bool(cluster_name)
            details.append({
                "object_id": entry["object_id"],
                "class_name": class_name,
                "cluster_id": int(label),
                "cluster_name": cluster_name,
                "assigned": assigned,
                "similarity": 0,
                "match_method": "cluster",
            })
            if assigned:
                assignments[entry["object_id"]] = cluster_name
                assigned_count += 1

        n_clusters = len(cluster_map)
        print(f"    Clusters: {n_clusters}, assigned: {n_clusters * min_samples}+")

    # Step 4: merge reference matches + cluster assignments
    assignments.update(match_assignments)
    details = match_details + details
    total = len(unlabeled)
    print(f"  Assigned: {len(assignments)}/{total}")
    return assignments, details
