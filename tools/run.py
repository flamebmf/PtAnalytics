#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.sync import api_get, api_post, cmd_backup, cmd_restore, cmd_train

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config():
    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def check_server(url):
    from urllib.request import urlopen, Request
    try:
        req = Request(f"{url.rstrip('/')}/objects", method="HEAD")
        with urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def backup(config):
    args = argparse.Namespace(url=config["server"]["url"], output=None)
    cmd_backup(args)


def restore(config):
    backup_path = input("  Путь к backup-файлу: ").strip()
    if not backup_path:
        print("  Отменено")
        return
    args = argparse.Namespace(url=config["server"]["url"], input=backup_path)
    cmd_restore(args)


def train(config):
    out_dir = config["paths"].get("training_dir", "./training")
    imgsz = config.get("training", {}).get("imgsz", 640)
    args = argparse.Namespace(url=config["server"]["url"], output_dir=out_dir, force=True, imgsz=imgsz)
    cmd_train(args)


def _generate_clip_report(out_dir):
    import base64, webbrowser
    from collections import defaultdict
    out = Path(out_dir)
    results_file = out / "auto_assign_results.json"
    details_file = out / "auto_assign_details.json"
    data_dir = out / "auto_assign_data"
    if not details_file.exists() or not data_dir.exists():
        print("  Нет данных. Сначала выполните Auto-assign.")
        return
    details = json.loads(details_file.read_text(encoding="utf-8"))
    if not details:
        print("  Нет назначений для отображения.")
        return
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    img_index = {}
    for entry in manifest.get("unlabeled", []):
        img_index[entry["object_id"]] = entry.get("crop") or entry.get("full")

    # Group by cluster_name (unassigned grouped separately)
    clusters = defaultdict(list)
    for d in details:
        key = d.get("cluster_name") or "__unassigned__"
        clusters[key].append(d)

    server_url = "http://192.168.5.12:8090"

    def _img_tag(oid):
        arcname = img_index.get(oid)
        if not arcname:
            return ""
        img_path = data_dir / arcname
        if not img_path.exists():
            return ""
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        # Find server image URL for this object
        server_img = ""
        for entry in manifest.get("unlabeled", []):
            if entry.get("object_id") == oid:
                sv = entry.get("server_img", "")
                if sv:
                    server_img = f'{server_url}/frames/{sv}'
                break
        link = f' <a href="{server_img}" target="_blank" title="Открыть на сервере" style="color:#0af;text-decoration:none">🔗</a>' if server_img else ""
        return f'<img src="data:image/jpeg;base64,{b64}" style="max-width:200px;border-radius:6px;margin:4px">{link}'

    sections = []
    assigned_total = 0
    # Assigned clusters first (sorted by name)
    cluster_keys = sorted(k for k in clusters if k != "__unassigned__")
    for key in cluster_keys:
        items = clusters[key]
        assigned_total += len(items)
        cards = "".join(
            f'<div style="display:inline-block;text-align:center;vertical-align:top;margin:6px">'
            f'{_img_tag(d["object_id"])}<br><code>{d["object_id"][:8]}..</code></div>'
            for d in items
        )
        sections.append(f'<h3 style="color:#0f0">✓ {key} ({len(items)})</h3><div>{cards}</div>')

    # Unassigned
    unassigned = clusters.get("__unassigned__", [])
    if unassigned:
        cards = "".join(
            f'<div style="display:inline-block;text-align:center;vertical-align:top;margin:6px;opacity:0.6">'
            f'{_img_tag(d["object_id"])}<br><code>{d["object_id"][:8]}..</code></div>'
            for d in unassigned
        )
        sections.append(f'<h3 style="color:#f80">✗ Не назначены ({len(unassigned)})</h3><div>{cards}</div>')

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>CLIP Auto-assign — Review</title>
<style>
body {{ font-family: sans-serif; background: #111; color: #eee; padding: 20px }}
h2 {{ color: #00d4ff }}
h3 {{ margin-top: 24px }}
code {{ font-size: 11px; color: #888 }}
</style></head><body>
<h2>CLIP Auto-assign — кластеры</h2>
<p>Назначено: <strong>{assigned_total}/{len(details)}</strong></p>
{''.join(sections)}</body></html>"""
    report_path = out / "clip_review.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  Отчёт: {report_path}")
    webbrowser.open(str(report_path))


def auto_assign(config):
    import io, zipfile, shutil
    from tools.auto_assign_clip import run_auto_assign

    url = config["server"]["url"]
    out_dir = Path(config["paths"].get("output_dir", "./tools-output"))
    eps = config["clip"].get("eps", 0.5)
    min_samples = config["clip"].get("min_samples", 2)
    clip_model = config["clip"].get("model", "ViT-L/14")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: backup
    print("  Creating backup...")
    backup_data, _ = api_get(url, "/backup")
    backup_path = out_dir / "pre_auto_assign_backup.zip"
    backup_path.write_bytes(backup_data)
    print(f"  Backup: {backup_path}")

    # Step 2: export data from server
    print("  Exporting data...")
    export_data, status = api_get(url, "/auto-assign/export")
    if status != 200:
        print("  Export failed")
        return

    # Step 3: extract
    extract_dir = out_dir / "auto_assign_data"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(io.BytesIO(export_data)) as zf:
        zf.extractall(str(extract_dir))

    manifest_path = extract_dir / "manifest.json"
    if not manifest_path.exists():
        print("  No manifest.json in export")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Step 4: run CLIP clustering (no upload)
    print("  Running CLIP clustering...")
    sim_threshold = config["clip"].get("sim_threshold", 0.85)
    assignments, details = run_auto_assign(
        extract_dir=extract_dir,
        manifest=manifest,
        eps=eps,
        min_samples=min_samples,
        sim_threshold=sim_threshold,
        clip_model=clip_model,
    )

    # Step 5: save results locally
    assign_path = out_dir / "auto_assign_results.json"
    assign_path.write_text(json.dumps(assignments, indent=2, ensure_ascii=False), encoding="utf-8")
    details_path = out_dir / "auto_assign_details.json"
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Назначено: {len(assignments)}/{len(details)} объектов")

    # Step 6: generate HTML report
    _generate_clip_report(out_dir)
    print("\n  Проверьте отчёт в браузере.")
    print('  Если всё ОК — загрузите: пункт "Upload CLIP results"')


def upload_clip(config):
    url = config["server"]["url"]
    out_dir = Path(config["paths"].get("output_dir", "./tools-output"))
    assign_path = out_dir / "auto_assign_results.json"
    if not assign_path.exists():
        print("  Нет результатов. Сначала выполните Auto-assign.")
        return
    assignments = json.loads(assign_path.read_text(encoding="utf-8"))
    if not assignments:
        print("  Нет назначений для загрузки.")
        return
    confirm = input(f"  Загрузить {len(assignments)} назначений на сервер? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Отменено")
        return
    data, status = api_post(url, "/auto-assign/upload", data={"assignments": assignments})
    result = json.loads(data)
    print(f"  Загружено: {result.get('assigned', 0)} объектов")


def review_clip(config):
    out_dir = config["paths"].get("output_dir", "./tools-output")
    _generate_clip_report(out_dir)


def deploy_model(config, silent=False):
    """Copy latest fine-tuned.pt to project training/ for docker-compose mount."""
    import shutil
    candidates = []
    for key in ("training_dir", "output_dir"):
        d = config["paths"].get(key)
        if d:
            p = Path(d) / "fine-tuned.pt"
            if p.exists():
                candidates.append((p.stat().st_mtime, p))
    if not candidates:
        if not silent:
            print("  Нет fine-tuned.pt ни в training_dir, ни в output_dir.")
        return
    candidates.sort(key=lambda x: x[0], reverse=True)
    src = candidates[0][1]
    dst = _PROJECT_ROOT / "training" / "fine-tuned.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    if not silent:
        print(f"  Скопирован: {src} → {dst}")
    # Also upload to server
    from tools.sync import api_post
    url = config["server"]["url"]
    data, status = api_post(url, "/training/upload", files=[
        ("model", "fine-tuned.pt", dst.read_bytes()),
    ])
    result = json.loads(data)
    if not silent:
        print(f"  Загружен на сервер: {result.get('status')}")


MENU = """
╔══════════════════════════════════════════════╗
║       PtAnalytics Tool Manager              ║
╠══════════════════════════════════════════════╣
║  1. Backup — скачать бэкап                  ║
║  2. Restore — восстановить из бэкапа        ║
║  3. Train YOLO — дообучить модель           ║
║  4. Auto-assign (CLIP) — назначить имена    ║
║  5. Review CLIP — посмотреть назначения     ║
║  6. Upload CLIP — загрузить на сервер       ║
║  7. Deploy model — скопировать модель в     ║
║     training/ и загрузить на сервер         ║
║  0. Выход                                   ║
╚══════════════════════════════════════════════╝
"""

ACTIONS = {
    "1": ("Backup", backup),
    "2": ("Restore", restore),
    "3": ("Train YOLO", train),
    "4": ("Auto-assign (CLIP)", auto_assign),
    "5": ("Review CLIP", review_clip),
    "6": ("Upload CLIP", upload_clip),
    "7": ("Deploy model", deploy_model),
}


def main():
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description="PtAnalytics Tool Manager")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config file")
    parser.add_argument("--cmd", choices=["backup", "restore", "train", "auto-assign", "upload-clip", "deploy"],
                        help="Run command directly without menu")
    args_known, _ = parser.parse_known_args()

    if args_known.config:
        CONFIG_PATH = Path(args_known.config)

    config = load_config()
    url = config["server"]["url"]

    print(f"Сервер: {url}")

    if args_known.cmd:
        cmd_map = {
            "backup": backup,
            "restore": restore,
            "train": train,
            "auto-assign": auto_assign,
            "upload-clip": upload_clip,
            "deploy": deploy_model,
        }
        cmd_map[args_known.cmd](config)
        return

    while True:
        print(MENU)
        choice = input("  Выберите действие (0-7): ").strip()

        if choice == "0":
            print("  Выход")
            break

        if choice in ACTIONS:
            name, func = ACTIONS[choice]
            if choice not in ("5", "6") and not check_server(url):
                print(f"  Сервер {url} недоступен. Запустите сначала сервер.")
                continue

            confirm = input(f"  Запустить '{name}'? (y/n): ").strip().lower()
            if confirm != "y":
                print("  Отменено")
                continue

            try:
                func(config)
                print(f"\n  ✓ {name} завершён")
            except Exception as e:
                print(f"\n  ✗ Ошибка: {e}")
        else:
            print("  Неверный ввод, попробуйте снова.")


if __name__ == "__main__":
    main()
