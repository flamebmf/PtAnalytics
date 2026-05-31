"""Train YOLO on exported datasets sequentially, save fine-tuned.pt."""
import sys, shutil, json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

BASE = Path(__file__).parent
EXTRACTED = BASE / "extracted"
MODELS = BASE / "models"
STATE_FILE = BASE / ".train-state.json"

IMGSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 640
BATCH = 2 if IMGSZ >= 1280 else 8
FORCE = "--force" in sys.argv

state = {}
if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text())

print(f"Training: imgsz={IMGSZ} batch={BATCH} force={FORCE}")
print(f"Already trained: {list(state.keys())}")

datasets = ["mazda", "lena", "im"]
current_model = "yolo11m.pt"

for name in datasets:
    yaml = EXTRACTED / name / "fine-tune-data" / "dataset.yaml"
    if not yaml.exists():
        print(f"SKIP {name}: dataset.yaml not found")
        continue

    if name in state and not FORCE:
        prev = state[name]
        print(f"SKIP {name}: already trained at {prev['date']} (mAP50={prev.get('mAP50','?')})")
        # Use the last best model for this dataset as starting point
        best_path = MODELS / name / "weights" / "best.pt"
        if best_path.exists():
            current_model = str(best_path)
        continue

    print(f"\n=== TRAINING {name} ===")
    model = YOLO(current_model)
    model.train(
        data=str(yaml),
        epochs=30,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,
        device=0,
        project=str(MODELS),
        name=name,
        exist_ok=True,
    )

    best = MODELS / name / "weights" / "best.pt"
    if best.exists():
        current_model = str(best)
        state[name] = {"date": datetime.now().isoformat(), "imgsz": IMGSZ}
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        print(f"DONE {name} — saved state")
        shutil.copy(str(best), str(BASE / f"fine-tuned-{name}.pt"))
    else:
        print(f"WARN {name} — best.pt not found")

final = MODELS / datasets[-1] / "weights" / "best.pt"
if final.exists():
    shutil.copy(str(final), str(BASE / "fine-tuned.pt"))
    print(f"\n=== FINAL MODEL: fine-tuned.pt ===")

