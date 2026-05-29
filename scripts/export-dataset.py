#!/usr/bin/env python3
"""
export-dataset.py — Export collected crops to YOLO-ready dataset.

Usage:
    python scripts/export-dataset.py [--crops /data/crops] [--output /data/dataset] [--val-split 0.2]

Output structure:
    /data/dataset/
        images/train/
        images/val/
        labels/train/
        labels/val/
        dataset.yaml
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export crops to YOLO dataset")
    parser.add_argument("--crops", default="/data/crops", help="Path to crop collection")
    parser.add_argument("--output", default="/data/dataset", help="Output dataset path")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    out_dir = Path(args.output)

    if not crops_dir.is_dir():
        print(f"Crops directory not found: {crops_dir}")
        return

    # Collect all class folders
    all_images = []
    class_names = set()

    for class_dir in sorted(crops_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_names.add(class_dir.name)
        for cam_dir in class_dir.iterdir():
            if not cam_dir.is_dir():
                continue
            for f in cam_dir.glob("*.jpg"):
                label = f.with_suffix(".txt")
                if label.exists():
                    all_images.append((f, label, class_dir.name))

    if not all_images:
        print(f"No crops found in {crops_dir}")
        return

    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - args.val_split))
    train = all_images[:split_idx]
    val = all_images[split_idx:]

    class_list = sorted(class_names)
    class_map = {name: idx for idx, name in enumerate(class_list)}

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in [("train", train), ("val", val)]:
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_path, cname in items:
            dst_img = img_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            dst_lbl = lbl_dir / label_path.name
            dst_lbl.write_text(f"{class_map[cname]} 0.5 0.5 1.0 1.0\n")

    # Write dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    yaml_content = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_list)}",
        f"names: {class_list}",
    ]
    yaml_path.write_text("\n".join(yaml_content) + "\n")

    print(f"Dataset exported to {out_dir}")
    print(f"  Classes: {class_list}")
    print(f"  Train: {len(train)} images")
    print(f"  Val:   {len(val)} images")


if __name__ == "__main__":
    main()
