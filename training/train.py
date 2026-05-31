"""Train YOLO on exported datasets sequentially, save fine-tuned.pt."""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent
EXTRACTED = BASE / "extracted"
MODELS = BASE / "models"

datasets = ["mazda", "lena", "im"]
current_model = "yolo11m.pt"

for name in datasets:
    yaml = EXTRACTED / name / "fine-tune-data" / "dataset.yaml"
    if not yaml.exists():
        print(f"SKIP {name}: {yaml} not found")
        continue

    print(f"\n=== TRAINING {name} ===")
    model = YOLO(current_model)
    model.train(
        data=str(yaml),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        device=0,
        project=str(MODELS),
        name=name,
        exist_ok=True,
    )

    best = MODELS / name / "weights" / "best.pt"
    if best.exists():
        current_model = str(best)
        print(f"DONE {name} — mAP50 saved to {best}")
        # Copy best to top-level after each step
        shutil.copy(str(best), str(BASE / f"fine-tuned-{name}.pt"))
    else:
        print(f"WARN {name} — best.pt not found, continuing with previous model")

final = MODELS / datasets[-1] / "weights" / "best.pt"
if final.exists():
    shutil.copy(str(final), str(BASE / "fine-tuned.pt"))
    print(f"\n=== FINAL MODEL: {BASE / 'fine-tuned.pt'} ===")
else:
    print(f"\n=== WARN: final model not found at {final} ===")
