import asyncio
import json
import os
import signal
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

MSK = timezone(timedelta(hours=3))

def _local(ts):
    """Convert naive UTC datetime to Moscow time isoformat."""
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc).astimezone(MSK).isoformat()

from aiohttp import web
from loguru import logger
from sqlalchemy import select, text

from .config import load_settings, load_cameras, load_triggers
from .pipeline import CameraPipeline
from .stats import StatsCollector
from .storage import init_db, close_db, init_pgvector, init_schema, StorageRepository, get_session
from .storage.models import TrackedObject, FrameCapture
from .actions import ActionDispatcher, MQTTAction


async def main():
    config_dir = Path(os.environ.get("CONFIG_DIR", "config"))
    settings = load_settings(config_dir)
    cameras_cfg = load_cameras(config_dir)
    triggers_cfg = load_triggers(config_dir)

    log_dir = Path(os.environ.get("LOG_DIR", "/app/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"cam-analyzer_{datetime.now().strftime('%Y%m%d')}.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.get("app", {}).get("log_level", "INFO"),
        colorize=True,
    )
    logger.add(
        log_file,
        level=settings.get("app", {}).get("log_level", "INFO"),
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}",
    )

    # --- DB ---
    db_cfg = settings.get("database", {})
    await init_db(
        host=os.environ.get("DB_HOST", db_cfg.get("host", "localhost")),
        port=int(os.environ.get("DB_PORT", db_cfg.get("port", 5432))),
        user=os.environ.get("DB_USER", db_cfg.get("user", "cam")),
        password=os.environ.get("DB_PASSWORD", db_cfg.get("password", "")),
        database=os.environ.get("DB_NAME", db_cfg.get("name", "cam")),
    )
    await init_pgvector()
    await init_schema()
    async with await get_session() as session:
        for stmt in [
            "ALTER TABLE tracked_objects ADD COLUMN IF NOT EXISTS name VARCHAR(128)",
            "ALTER TABLE tracked_objects ADD COLUMN IF NOT EXISTS ignored BOOLEAN DEFAULT FALSE",
        ]:
            try:
                await session.execute(text(stmt))
            except Exception:
                pass
        await session.commit()

    data_dir = Path(settings.get("app", {}).get("data_dir", "/data/frames"))
    repository = StorageRepository(data_dir=data_dir)

    # --- MQTT ---
    mqtt_cfg = settings.get("mqtt", {})
    MQTTAction.configure(
        host=os.environ.get("MQTT_HOST", mqtt_cfg.get("host", "localhost")),
        port=int(os.environ.get("MQTT_PORT", mqtt_cfg.get("port", 1883))),
        client_id=mqtt_cfg.get("client_id", "cam-analyzer"),
    )

    dispatcher = ActionDispatcher()
    dispatcher.load_triggers(triggers_cfg.get("triggers", []))

    cameras = cameras_cfg.get("cameras", [])
    if not cameras:
        logger.error("No cameras configured")
        return

    # --- Stats ---
    stats_collector = StatsCollector(log_interval=30.0)

    # --- Pipelines ---
    pipelines: list[CameraPipeline] = []
    tasks: list[asyncio.Task] = []

    for cam_cfg in cameras:
        pipeline = CameraPipeline(
            camera_config=cam_cfg,
            settings=settings,
            repository=repository,
            dispatcher=dispatcher,
        )
        pipelines.append(pipeline)
        stats_collector._pipelines[pipeline.cam_id] = pipeline.stats
        tasks.append(asyncio.create_task(pipeline.run()))

    logger.info(f"Started {len(tasks)} camera pipelines")

    # --- Stats log background task ---
    async def stats_loop():
        while True:
            await asyncio.sleep(30)
            stats_collector.maybe_log()

    tasks.append(asyncio.create_task(stats_loop()))

    logger.info("Starting HTTP health server...")

    # --- Health HTTP server ---
    health_port = int(os.environ.get("HEALTH_PORT", "8090"))
    health_app = web.Application()

    async def handle_health(request: web.Request) -> web.Response:
        db_ok = False
        try:
            async with await get_session() as session:
                await session.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            pass

        summary = stats_collector.summary()
        running = summary["cameras_running"]
        if running > 0 and db_ok:
            status = "healthy"
        elif running > 0:
            status = "degraded"
        else:
            status = "unhealthy"

        return web.json_response({"status": status, "db_connected": db_ok, **summary})

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response(stats_collector.summary())

    async def handle_detailed(request: web.Request) -> web.Response:
        return web.json_response(stats_collector.snapshot())

    async def handle_list_objects(request: web.Request) -> web.Response:
        camera_id = request.query.get("camera_id")
        class_name = request.query.get("class_name")
        name = request.query.get("name")
        show_ignored = request.query.get("show_ignored") == "1"
        limit = int(request.query.get("limit", 50))
        offset = int(request.query.get("offset", 0))
        objects = await repository.list_objects(
            camera_id=camera_id, class_name=class_name, name=name, show_ignored=show_ignored,
            limit=limit, offset=offset,
        )
        total = await repository.get_object_count(
            camera_id=camera_id, class_name=class_name, show_ignored=show_ignored,
        )
        items = [
            {
                "id": str(obj.id),
                "camera_id": obj.camera_id,
                "track_id": obj.track_id,
                "class_name": obj.class_name,
                "name": obj.name,
                "ignored": obj.ignored,
                "plate_number": obj.plate_number,
                "face_id": obj.face_id,
                "first_seen": _local(obj.first_seen),
                "last_seen": _local(obj.last_seen),
                "appearance_count": obj.appearance_count,
            }
            for obj in objects
        ]
        return web.json_response({"total": total, "limit": limit, "offset": offset, "items": items})

    async def handle_get_object(request: web.Request) -> web.Response:
        obj_id = uuid.UUID(request.match_info["id"])
        async with await get_session() as session:
            result = await session.execute(select(TrackedObject).where(TrackedObject.id == obj_id))
            obj = result.scalar_one_or_none()
            if obj is None:
                return web.json_response({"error": "not found"}, status=404)
        frames = await repository.list_frames(object_id=obj_id)
        return web.json_response({
            "id": str(obj.id),
            "camera_id": obj.camera_id,
            "track_id": obj.track_id,
            "class_name": obj.class_name,
            "name": obj.name,
            "ignored": obj.ignored,
            "plate_number": obj.plate_number,
            "face_id": obj.face_id,
            "first_seen": _local(obj.first_seen),
            "last_seen": _local(obj.last_seen),
            "appearance_count": obj.appearance_count,
            "frames": [
                {
                    "id": str(f.id),
                    "image": f"/frames/{Path(f.image_path).name}",
                    "bbox": [f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2],
                    "confidence": f.confidence,
                    "timestamp": _local(f.timestamp),
                }
                for f in frames
            ],
        })

    async def handle_patch_object(request: web.Request) -> web.Response:
        obj_id = uuid.UUID(request.match_info["id"])
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        name = body.get("name")
        if name is not None and not isinstance(name, str):
            return web.json_response({"error": "name must be a string"}, status=400)
        if name == "":
            name = None
        ok = await repository.update_object_name(obj_id, name)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "updated"})

    async def handle_delete_object(request: web.Request) -> web.Response:
        obj_id = uuid.UUID(request.match_info["id"])
        ok = await repository.delete_object(obj_id)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "deleted"})

    async def handle_delete_frame(request: web.Request) -> web.Response:
        frame_id = uuid.UUID(request.match_info["id"])
        ok = await repository.delete_frame(frame_id)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "deleted"})

    async def handle_get_frame(request: web.Request) -> web.Response:
        filename = request.match_info["filename"]
        if ".." in filename or "/" in filename or "\\" in filename:
            return web.json_response({"error": "invalid path"}, status=400)
        frame_dir = Path(settings.get("app", {}).get("data_dir", "/data/frames"))
        filepath = frame_dir / filename
        if not filepath.exists() or not filepath.is_file():
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(filepath)

    async def handle_ui(request: web.Request) -> web.Response:
        ui_file = Path(__file__).resolve().parent.parent / "ui" / "index.html"
        if not ui_file.exists():
            return web.json_response({"error": "UI not found"}, status=404)
        return web.FileResponse(ui_file)

    health_app.router.add_get("/", handle_ui)
    health_app.router.add_get("/ui", handle_ui)
    health_app.router.add_get("/health", handle_health)
    health_app.router.add_get("/stats", handle_stats)
    health_app.router.add_get("/stats/detailed", handle_detailed)
    async def handle_list_names(request: web.Request) -> web.Response:
        names = await repository.list_object_names()
        return web.json_response({"names": names})

    async def handle_ignore_object(request):
        obj_id = uuid.UUID(request.match_info["id"])
        changed = False
        async with await get_session() as session:
            result = await session.execute(select(TrackedObject).where(TrackedObject.id == obj_id))
            obj = result.scalar_one_or_none()
            if obj is None:
                return web.json_response({"error": "not found"}, status=404)
            if not obj.ignored:
                obj.ignored = True
                changed = True
                await session.commit()
        # Also mark the in-memory active track as ignored
        if changed:
            for pl in pipelines:
                pl.mark_track_ignored(camera_id=obj.camera_id, track_id=obj.track_id)
        return web.json_response({"ok": True, "ignored": True})

    health_app.router.add_get("/objects", handle_list_objects)
    health_app.router.add_get("/objects/names", handle_list_names)
    health_app.router.add_get("/objects/{id}", handle_get_object)
    health_app.router.add_patch("/objects/{id}", handle_patch_object)
    health_app.router.add_delete("/objects/{id}", handle_delete_object)
    health_app.router.add_post("/objects/{id}/ignore", handle_ignore_object)
    health_app.router.add_delete("/frames/{id}", handle_delete_frame)
    health_app.router.add_get("/frames/{filename:.+}", handle_get_frame)

    runner = web.AppRunner(health_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", health_port)
    await site.start()
    logger.info(f"Health endpoint: http://0.0.0.0:{health_port}/health")

    # --- Shutdown ---
    loop = asyncio.get_running_loop()

    def shutdown():
        logger.info("Shutting down...")
        for pl in pipelines:
            pl.stop()
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        await MQTTAction.disconnect()
        await close_db()
        logger.info("Shutdown complete")


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
