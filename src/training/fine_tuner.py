# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
import asyncio
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from loguru import logger
from sqlalchemy import select, func

from ..storage.db import get_session
from ..storage.models import TrackedObject, FrameCapture


class FineTuner:
    """Collects crops from user-named objects and fine-tunes YOLO periodically."""

    def __init__(
        self,
        models_dir: Path,
        data_dir: Path,
        config: dict,
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.cfg = config.get("training", {})
        self.enabled = self.cfg.get("enabled", False)
        self.min_samples = self.cfg.get("min_samples_per_class", 30)
        self.epochs = self.cfg.get("epochs", 30)
        self.imgsz = self.cfg.get("imgsz", 1280)
        self.base_model = self.cfg.get("base_model", "yolo11m.pt")
        cfg_device = config.get("detector", {}).get("device", "cpu")
        if cfg_device == "cpu" and self._cuda_available():
            cfg_device = "cuda:0"
        self.device = cfg_device
        self.workers = config.get("detector", {}).get("workers")
        self.batch_size = self.cfg.get("batch_size", 8)
        self.min_show = self.cfg.get("min_show_frames", 5)
        self.net = None

    @property
    def dataset_root(self) -> Path:
        return self.models_dir / "fine-tune-data"

    @property
    def output_model(self) -> Path:
        return self.models_dir / "fine-tuned.pt"

    @property
    def state_file(self) -> Path:
        return self.models_dir / ".fine-tune-state"

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _read_state(self) -> dict:
        if self.state_file.exists():
            return yaml.safe_load(self.state_file.read_text()) or {}
        return {}

    def _write_state(self, state: dict):
        self.state_file.write_text(yaml.dump(state))

    def _find_base_model(self) -> str:
        yolo_dir = os.environ.get("YOLO_CONFIG_DIR", str(self.models_dir))
        for d in [yolo_dir, str(self.models_dir),
                  os.path.expanduser("~/.config/ultralytics"),
                  "/app/models/ultralytics", "/app/models"]:
            if not d:
                continue
            p = os.path.join(d, self.base_model)
            if os.path.isfile(p):
                return p
        return os.path.join(yolo_dir, self.base_model)

    async def count_labeled_samples(self) -> int:
        async with await get_session() as session:
            result = await session.execute(
                select(func.count(FrameCapture.id))
                .join(TrackedObject, FrameCapture.object_id == TrackedObject.id)
                .where(TrackedObject.name.isnot(None))
                .where(TrackedObject.name != "")
                .where(TrackedObject.ignored != True)
            )
            return result.scalar() or 0

    async def collect_dataset(self, name_filter: Optional[str] = None) -> Optional[Path]:
        """Export YOLO dataset from named objects. Optionally filter by object name."""
        async with await get_session() as session:
            q = (
                select(TrackedObject)
                .where(TrackedObject.name.isnot(None))
                .where(TrackedObject.name != "")
                .where(TrackedObject.ignored != True)
            )
            if name_filter:
                q = q.where(TrackedObject.name == name_filter)
            result = await session.execute(q)
            named_objects = result.scalars().all()

        if not named_objects:
            logger.info("FineTune: no named objects found")
            return None

        if name_filter:
            # Per-name training: use dominant class, merge all frames regardless of current class
            group_key = "class_name"
            class_groups: dict[str, list[TrackedObject]] = {}
            for obj in named_objects:
                class_groups.setdefault(obj.class_name, []).append(obj)
            best_cls, best_objs, best_total = None, [], 0
            for cls_name, objs in class_groups.items():
                total = await self._count_frames_for_objects(objs)
                if total > best_total:
                    best_cls, best_objs, best_total = cls_name, objs, total
            if best_total < self.min_samples:
                logger.info(f"FineTune: '{name_filter}' has only {best_total} frames in dominant class '{best_cls}' (need {self.min_samples})")
                return None
            valid = {best_cls: named_objects}
            logger.info(f"FineTune: '{name_filter}' → class '{best_cls}' with {best_total} frames ({len(named_objects)} objects)")
        else:
            # Combined training: group by user-assigned NAME, each name = one class
            class_groups: dict[str, list[TrackedObject]] = {}
            for obj in named_objects:
                class_groups.setdefault(obj.name, []).append(obj)
            valid = {}
            for name, objs in class_groups.items():
                total = await self._count_frames_for_objects(objs)
                if total >= self.min_samples:
                    valid[name] = objs
                    logger.info(f"FineTune: name '{name}' has {total} frames from {len(objs)} objects")

        if not valid:
            logger.info("FineTune: no class has enough samples (need {})", self.min_samples)
            return None

        dataset_dir = self.dataset_root
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        classes_yaml = list(valid.keys())

        for cls_name, objs in valid.items():
            img_dir = dataset_dir / "train" / "images"
            lbl_dir = dataset_dir / "train" / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            cls_idx = classes_yaml.index(cls_name)
            saved = 0

            async with await get_session() as session:
                for obj in objs:
                    frames_result = await session.execute(
                        select(FrameCapture)
                        .where(FrameCapture.object_id == obj.id)
                        .order_by(FrameCapture.timestamp)
                    )
                    for fc in frames_result.scalars():
                        crop = self._read_crop(fc)
                        if crop is None:
                            continue
                        img_name = f"{cls_name}_{saved:06d}.jpg"
                        img_path = img_dir / img_name
                        cv2.imwrite(str(img_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        label_path = lbl_dir / f"{cls_name}_{saved:06d}.txt"
                        label_path.write_text(f"{cls_idx} 0.5 0.5 1.0 1.0\n")
                        saved += 1

            logger.info(f"FineTune: class '{cls_name}' exported {saved} crops")

        # Validation split (10%)
        for cls_name in classes_yaml:
            img_dir = dataset_dir / "train" / "images"
            lbl_dir = dataset_dir / "train" / "labels"
            val_img = dataset_dir / "val" / "images"
            val_lbl = dataset_dir / "val" / "labels"
            val_img.mkdir(parents=True, exist_ok=True)
            val_lbl.mkdir(parents=True, exist_ok=True)
            prefix = f"{cls_name}_"
            images = sorted([f for f in os.listdir(str(img_dir)) if f.startswith(prefix)])
            val_count = max(1, len(images) // 10)
            for img_name in images[-val_count:]:
                shutil.move(str(img_dir / img_name), str(val_img / img_name))
                lbl_name = img_name.replace(".jpg", ".txt")
                shutil.move(str(lbl_dir / lbl_name), str(val_lbl / lbl_name))

        dataset_yaml = {
            "path": ".",
            "train": "train/images",
            "val": "val/images",
            "nc": len(classes_yaml),
            "names": classes_yaml,
        }
        yaml_path = dataset_dir / "dataset.yaml"
        yaml_path.write_text(yaml.dump(dataset_yaml, default_flow_style=False))
        logger.info(f"FineTune: dataset ready at {dataset_dir}")
        return dataset_dir

    async def train(self, dataset_dir: Path, epoch_callback=None) -> Optional[str]:
        """Run YOLO fine-tuning. Returns model path on success."""
        from ultralytics import YOLO
        base_path = self._find_base_model()
        yaml_path = dataset_dir / "dataset.yaml"

        if not yaml_path.exists():
            logger.error(f"FineTune: dataset.yaml not found at {yaml_path}")
            return None

        logger.info(f"FineTune: starting training from {base_path}, {self.epochs} epochs")
        loop = asyncio.get_running_loop()
        cb = epoch_callback

        def _train():
            import sys, io
            model = YOLO(base_path)

            current_epoch = [0]
            current_total = [self.epochs]

            def on_train_epoch_end(trainer):
                current_epoch[0] = trainer.epoch + 1
                current_total[0] = trainer.epochs

            def on_val_end(trainer):
                ep = current_epoch[0]
                total = current_total[0]
                mp, mr = 0.0, 0.0
                try:
                    if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                        res = trainer.metrics.mean_results()
                        if len(res) >= 4:
                            mp, mr = float(res[2]), float(res[3])
                except Exception:
                    pass
                logger.info(f"FineTune epoch {ep}/{total} | mAP50={mp:.4f} mAP50-95={mr:.4f}")
                if cb:
                    cb(ep, total, 0, 0, 0)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_val_end", on_val_end)

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                model.train(
                    data=str(yaml_path),
                    epochs=self.epochs,
                    imgsz=self.imgsz,
                    device=self.device,
                    workers=self.workers or 4,
                    batch=self.batch_size,
                    exist_ok=True,
                    project=str(self.models_dir / "runs"),
                    name="fine-tune",
                    verbose=False,
                )
            finally:
                captured = sys.stdout.getvalue()
                sys.stdout = old_stdout
                # Log YOLO summary lines only (model summary, dataset scan, optimizer)
                for line in captured.split('\n'):
                    stripped = line.strip()
                    if stripped and any(kw in stripped for kw in ('model.yaml', 'summary', 'train:', 'val:', 'optimizer:', 'Starting training', 'Image sizes')):
                        logger.info(f"YOLO: {stripped}")
            best_pt = self.models_dir / "runs" / "fine-tune" / "weights" / "best.pt"
            if best_pt.exists():
                shutil.copy(str(best_pt), str(self.output_model))
                logger.info(f"FineTune: best model saved to {self.output_model}")
                return str(self.output_model)
            return None

        try:
            result = await loop.run_in_executor(None, _train)
            if result:
                state = self._read_state()
                state["last_train"] = datetime.now(timezone.utc).isoformat()
                state["model"] = str(result)
                total = await self.count_labeled_samples()
                state["samples_at_train"] = total
                self._write_state(state)
            return result
        except Exception as e:
            logger.error(f"FineTune: training failed: {e}")
            return None

    async def list_candidates(self) -> list[dict]:
        """Return named objects grouped by name with per-name sample counts."""
        async with await get_session() as session:
            named = await session.execute(
                select(TrackedObject)
                .where(TrackedObject.name.isnot(None))
                .where(TrackedObject.name != "")
                .where(TrackedObject.ignored != True)
            )
            objs = named.scalars().all()
        groups: dict[str, list[TrackedObject]] = {}
        for obj in objs:
            groups.setdefault(obj.name, []).append(obj)
        result = []
        for name, name_objs in sorted(groups.items()):
            classes = list(dict.fromkeys(o.class_name for o in name_objs))
            total = await self._count_frames_for_objects(name_objs)
            if total < self.min_show:
                continue
            result.append({
                "name": name,
                "class_name": classes[0] if len(classes) == 1 else ", ".join(classes),
                "objects": len(name_objs),
                "frames": total,
                "ready": total >= self.min_samples,
            })
        result.sort(key=lambda x: x["frames"], reverse=True)
        return result

    async def _count_frames_for_objects(self, objects: list[TrackedObject]) -> int:
        ids = [obj.id for obj in objects]
        async with await get_session() as session:
            result = await session.execute(
                select(func.count(FrameCapture.id))
                .where(FrameCapture.object_id.in_(ids))
            )
            return result.scalar() or 0

    async def export_zip(self, name_filter: Optional[str] = None) -> Optional[Path]:
        """Export YOLO dataset as ZIP file. Returns path to zip."""
        dataset_dir = await self.collect_dataset(name_filter=name_filter)
        if dataset_dir is None:
            return None
        name = name_filter or "all"
        zip_path = dataset_dir.parent / f"dataset-{name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._create_zip, dataset_dir, zip_path)
        return zip_path

    @staticmethod
    def _create_zip(dataset_dir: Path, zip_path: Path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in dataset_dir.rglob('*'):
                if f.is_file():
                    zf.write(f, f.relative_to(dataset_dir.parent))

    @staticmethod
    def _read_crop(fc: FrameCapture) -> Optional[np.ndarray]:
        """Extract clean crop from saved frame image using bbox."""
        try:
            img_path = Path(fc.image_path)
            if not img_path.exists():
                return None
            frame = cv2.imread(str(img_path))
            if frame is None:
                return None
            h, w = frame.shape[:2]
            x1 = max(0, fc.bbox_x1)
            y1 = max(0, fc.bbox_y1)
            x2 = min(w, fc.bbox_x2)
            y2 = min(h, fc.bbox_y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return frame[y1:y2, x1:x2]
        except Exception:
            return None
