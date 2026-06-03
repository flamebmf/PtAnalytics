# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
"""
CLI tool for syncing with a remote PtAnalytics server.
Supports backup/restore, training, and auto-assignment.

Usage:
  python tools/sync.py backup --url http://server:8080
  python tools/sync.py restore --url http://server:8080 --input backup.zip
  python tools/sync.py train --url http://server:8080 [--output-dir ./training]
  python tools/sync.py auto-assign --url http://server:8080 [--threshold 0.85] [--output-dir ./tools-output]
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode


def api_get(url, path):
    full = url.rstrip("/") + path
    with urlopen(full) as r:
        return r.read(), r.status


def api_post(url, path, data=None, headers=None, files=None):
    import http.client
    full = url.rstrip("/") + path
    if files:
        import uuid
        boundary = "----" + uuid.uuid4().hex
        body = b""
        for name, filename, content in files:
            body += b"--" + boundary.encode() + b"\r\n"
            body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            body += b"Content-Type: application/octet-stream\r\n\r\n"
            body += content + b"\r\n"
        body += b"--" + boundary.encode() + b"--\r\n"
        req = Request(full, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    elif isinstance(data, dict):
        body = json.dumps(data).encode()
        req = Request(full, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = Request(full, method="POST")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urlopen(req) as r:
        return r.read(), r.status


# ── Commands ──────────────────────────────────────────────────────────

def cmd_backup(args):
    print(f"Downloading backup from {args.url}...")
    data, status = api_get(args.url, "/backup")
    if status != 200:
        print(f"Error: HTTP {status}")
        sys.exit(1)
    path = args.output or f"backup_{Path(args.url).name or 'server'}.zip"
    Path(path).write_bytes(data)
    print(f"Backup saved to {path}")


def cmd_restore(args):
    print(f"Restoring from {args.input}...")
    with open(args.input, "rb") as f:
        content = f.read()
    data, status = api_post(args.url, "/backup/restore", files=[
        ("file", os.path.basename(args.input), content),
    ])
    result = json.loads(data)
    print(f"Restored {result.get('restored', 0)} objects")


def cmd_train(args):
    out_dir = Path(args.output_dir or "training")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: backup
    print("Creating backup...")
    backup_data, _ = api_get(args.url, "/backup")
    backup_path = out_dir / "pre_train_backup.zip"
    backup_path.write_bytes(backup_data)
    print(f"Backup saved to {backup_path}")

    # Step 2: export dataset
    print("Exporting dataset...")
    zip_data, status = api_get(args.url, "/training/export")
    if status == 404:
        print("Not enough samples — nothing to train")
        sys.exit(1)
    dataset_zip = out_dir / "dataset.zip"
    dataset_zip.write_bytes(zip_data)
    print(f"Dataset saved to {dataset_zip}")

    # Step 3: run local training
    print("Running training (this may take a while)...")
    train_script = Path(__file__).resolve().parent / "train_yolo.py"
    if not train_script.exists():
        train_script = Path(__file__).resolve().parent / "train_yolo.bat"
    result = subprocess.run(
        [sys.executable, str(train_script), str(dataset_zip)],
        cwd=str(out_dir),
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Training failed:", result.stderr)
        sys.exit(1)

    # Step 4: upload model
    model_path = out_dir / "fine-tuned.pt"
    if not model_path.exists():
        print("fine-tuned.pt not found after training")
        sys.exit(1)
    print("Uploading fine-tuned model...")
    data, status = api_post(args.url, "/training/upload", files=[
        ("model", "fine-tuned.pt", model_path.read_bytes()),
    ])
    result = json.loads(data)
    print(f"Model uploaded: {result.get('status')}")


def cmd_auto_assign(args):
    out_dir = Path(args.output_dir or "tools-output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: backup
    print("Creating backup...")
    backup_data, _ = api_get(args.url, "/backup")
    backup_path = out_dir / "pre_auto_assign_backup.zip"
    backup_path.write_bytes(backup_data)
    print(f"Backup saved to {backup_path}")

    # Step 2: export
    print("Exporting data from server...")
    zip_data, status = api_get(args.url, "/auto-assign/export")
    if status != 200:
        print("Export failed")
        sys.exit(1)
    export_zip = out_dir / "auto_assign_export.zip"
    export_zip.write_bytes(zip_data)

    # Step 3: extract
    extract_dir = out_dir / "auto_assign_data"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(export_zip) as zf:
        zf.extractall(str(extract_dir))

    manifest_path = extract_dir / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json in export")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())

    # Step 4: run CLIP
    print("Running CLIP auto-assignment...")
    from tools.auto_assign_clip import run_auto_assign
    assignments = run_auto_assign(
        extract_dir=extract_dir,
        manifest=manifest,
        threshold=args.threshold,
    )

    if not assignments:
        print("No assignments found above threshold")
        sys.exit(0)

    # Step 5: upload results
    print(f"Uploading {len(assignments)} assignments...")
    data, status = api_post(args.url, "/auto-assign/upload", data={
        "assignments": assignments,
    })
    result = json.loads(data)
    print(f"Assigned {result.get('assigned', 0)} objects")

    # Save assignments locally too
    assign_path = out_dir / "auto_assign_results.json"
    assign_path.write_text(json.dumps(assignments, indent=2, ensure_ascii=False))
    print(f"Assignments saved to {assign_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PtAnalytics sync tool")
    parser.add_argument("command", choices=["backup", "restore", "train", "auto-assign"])
    parser.add_argument("--url", required=True, help="Server URL (e.g. http://192.168.1.100:8080)")
    parser.add_argument("--input", help="Input file (for restore)")
    parser.add_argument("--output", help="Output file (for backup)")
    parser.add_argument("--output-dir", help="Working directory")
    parser.add_argument("--threshold", type=float, default=0.85, help="CLIP similarity threshold")
    args = parser.parse_args()

    if args.command == "backup":
        cmd_backup(args)
    elif args.command == "restore":
        cmd_restore(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "auto-assign":
        cmd_auto_assign(args)


if __name__ == "__main__":
    main()
