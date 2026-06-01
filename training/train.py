# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
"""Train YOLO on combined dataset, produce fine-tuned.pt with all custom classes."""
import sys, shutil, json, torch, yaml, zipfile
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

BASE = Path(__file__).parent
MODELS = BASE / "models"
STATE_FILE = BASE / ".train-state.json"

IMGSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 640
BATCH = 2 if IMGSZ >= 1280 else 8
FORCE = "--force" in sys.argv

BASE_MODEL = "yolo11m.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
state = {}
if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text())

# Auto-extract ZIP if present (always overwrites existing folder)
dataset_dir = BASE / "extracted" / "combined" / "fine-tune-data"
dataset_yaml = dataset_dir / "dataset.yaml"
extracted_dir = BASE / "extracted"
zips = list(extracted_dir.glob("*.zip")) + list(BASE.glob("combined*.zip"))
if zips:
    zip_path = zips[0]
    print(f"Found ZIP: {zip_path}, extracting...")
    target = extracted_dir / "combined"
    shutil.rmtree(target, ignore_errors=True)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(target))
    print(f"Extracted {len(z.namelist())} files")
if not dataset_yaml.exists():
    print(f"ERROR: dataset.yaml not found at {dataset_yaml}")
    print("Run export via UI first: обучить все классы вместе")
    sys.exit(1)

# Fix dataset.yaml path to actual location on this machine
with open(str(dataset_yaml), "r") as f:
    data_cfg = yaml.safe_load(f)
data_cfg["path"] = str(dataset_dir)
with open(str(dataset_yaml), "w") as f:
    yaml.dump(data_cfg, f, default_flow_style=False)
print(f"Dataset path fixed: {dataset_dir}")

last_train = state.get("combined", {})
if not FORCE and last_train.get("date") and last_train.get("imgsz") == IMGSZ:
    print(f"SKIP: combined already trained at {last_train['date']} imgsz={IMGSZ}")
    print("Use --force to re-train")
    sys.exit(0)

print(f"Training combined model: imgsz={IMGSZ} batch={BATCH}")
model = YOLO(BASE_MODEL)
print(f"Device: {DEVICE}")
try:
    model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,
        device=DEVICE,
        project=str(MODELS),
        name="combined",
        exist_ok=True,
        patience=30,
    )
except Exception as e:
    print(f"Training exit (best.pt should be saved already): {e}")

best = MODELS / "combined" / "weights" / "best.pt"
if best.exists():
    dst = str(BASE / "fine-tuned.pt")
    try:
        shutil.copy(str(best), dst)
    except Exception as e:
        print(f"WARNING: copy to {dst} failed: {e}")
        dst2 = str(BASE / "fine-tuned_new.pt")
        try:
            shutil.copy(str(best), dst2)
            print(f"Saved as {dst2} instead")
        except Exception as e2:
            print(f"ERROR: copy to {dst2} also failed: {e2}")
    state["combined"] = {"date": datetime.now().isoformat(), "imgsz": IMGSZ}
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    # Clean up old per-name artifacts
    for name_dir in ["im", "lena", "mazda"]:
        p = MODELS / name_dir
        if p.exists():
            shutil.rmtree(p)
            print(f"Cleaned: models/{name_dir}")
    for old_pt in BASE.glob("fine-tuned-*.pt"):
        old_pt.unlink()
        print(f"Cleaned: {old_pt.name}")
    print(f"DONE: fine-tuned.pt saved")
else:
    print("ERROR: best.pt not found")
