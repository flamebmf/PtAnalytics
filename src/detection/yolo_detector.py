import os
import numpy as np
import torch
from ultralytics import YOLO
from typing import Optional
from loguru import logger


class YoloDetector:
    """YOLO-based object detector with configurable backend."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "cpu",
        confidence: float = 0.4,
        iou: float = 0.45,
        classes: Optional[list[int]] = None,
        imgsz: int = 640,
        workers: int | None = None,
        backend: str = "torch",
        min_bbox_size: int = 40,
    ):
        if workers:
            os.environ.setdefault("OMP_NUM_THREADS", str(workers))
            os.environ.setdefault("MKL_NUM_THREADS", str(workers))
            torch.set_num_threads(workers)

        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.classes = classes or [0, 1, 2, 3, 5, 7]
        self.imgsz = imgsz
        self.backend = backend
        self.min_bbox_size = min_bbox_size

        local_path = self._find_model(model_path)
        self.model = self._load_model(local_path)
        logger.info(f"YOLO: {os.path.basename(local_path)} device={device} imgsz={imgsz} workers={workers or 'default'} backend={backend}")

    def _find_model(self, name: str) -> str:
        if os.path.isfile(name):
            return os.path.abspath(name)
        yolo_dir = os.environ.get("YOLO_CONFIG_DIR", "")
        search = [yolo_dir] if yolo_dir else []
        for d in search + [
            os.path.join(os.environ.get("YOLO_CONFIG_DIR", "/app/models"), ".."),
            "/app/models/ultralytics", "/app/models",
            os.path.expanduser("~/.config/ultralytics"),
        ]:
            if not d:
                continue
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return os.path.abspath(p)
        return os.path.join(yolo_dir or "/app/models", name)

    def _load_model(self, model_path: str) -> YOLO:
        if self.backend == "openvino":
            return self._load_openvino(model_path)
        return YOLO(model_path)

    def _ov_dir(self, model_path: str) -> str:
        base = os.path.splitext(os.path.basename(model_path))[0]
        return os.path.join(os.path.dirname(model_path), f"{base}_openvino_model")

    def _load_openvino(self, model_path: str) -> YOLO:
        ov_path = self._ov_dir(model_path)
        if os.path.isdir(ov_path):
            xml_file = os.path.join(ov_path, f"{os.path.basename(ov_path).replace('_openvino_model', '')}.xml")
            bin_file = xml_file.replace(".xml", ".bin")
            valid = os.path.isfile(xml_file) and os.path.isfile(bin_file) and os.path.getsize(bin_file) > 1000
            if valid:
                logger.info(f"Loading OpenVINO model from {ov_path}")
                return YOLO(ov_path)
            logger.warning(f"Corrupt OpenVINO model at {ov_path}, re-exporting...")
            import shutil
            shutil.rmtree(ov_path)
        logger.info(f"Exporting {model_path} to OpenVINO (one-time)...")
        tmp = YOLO(model_path, task="detect")
        tmp.export(format="openvino", imgsz=self.imgsz, half=False)
        logger.info(f"OpenVINO export done, loading from {ov_path}")
        return YOLO(ov_path, task="detect")

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(
            frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections = []
        min_sz = self.min_bbox_size
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                if w < min_sz or h < min_sz:
                    continue
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                })
        return detections

    @property
    def class_names(self) -> dict:
        return self.model.names
