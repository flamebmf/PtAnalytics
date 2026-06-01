# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
import json
from typing import Any

from loguru import logger

from .base import BaseAction
from ..storage.models import TrackedObject


class MQTTAction(BaseAction):
    """Publish event to MQTT topic."""

    _client = None
    _host: str = ""
    _port: int = 1883
    _client_id: str = ""

    @classmethod
    def configure(cls, host: str, port: int, client_id: str):
        cls._host = host
        cls._port = port
        cls._client_id = client_id

    @classmethod
    async def _ensure_connected(cls):
        if cls._client is not None:
            return
        try:
            import paho.mqtt.client as mqtt
            import asyncio
            cls._client = mqtt.Client(client_id=cls._client_id)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: cls._client.connect(cls._host, cls._port, keepalive=60))
            cls._client.loop_start()
            logger.info(f"MQTT connected to {cls._host}:{cls._port}")
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            cls._client = None

    @classmethod
    async def disconnect(cls):
        if cls._client:
            cls._client.loop_stop()
            cls._client.disconnect()
            cls._client = None

    def __init__(self, topic: str, qos: int = 1):
        self.topic = topic
        self.qos = qos

    async def execute(self, obj: TrackedObject, trigger_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_connected()
        if self._client is None:
            return {"error": "MQTT not connected"}

        payload = json.dumps({
            "trigger": trigger_name,
            "object_id": str(obj.id),
            "camera_id": obj.camera_id,
            "class": obj.class_name,
            "plate_number": obj.plate_number,
            "face_id": obj.face_id,
            "timestamp": str(obj.last_seen),
            "metadata": metadata,
        })

        result = self._client.publish(self.topic, payload, qos=self.qos)
        logger.debug(f"MQTT published to {self.topic}: rc={result.rc}")
        return {"topic": self.topic, "rc": result.rc, "mid": result.mid}
