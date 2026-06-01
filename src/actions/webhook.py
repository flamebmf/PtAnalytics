# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
import json
from typing import Any

import aiohttp
from loguru import logger

from .base import BaseAction
from ..storage.models import TrackedObject


class WebhookAction(BaseAction):
    """Send HTTP request on trigger."""

    def __init__(self, url: str, method: str = "POST", headers: dict | None = None):
        self.url = url
        self.method = method.upper()
        self.headers = headers or {"Content-Type": "application/json"}

    async def execute(self, obj: TrackedObject, trigger_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "trigger": trigger_name,
            "object_id": str(obj.id),
            "camera_id": obj.camera_id,
            "class": obj.class_name,
            "plate_number": obj.plate_number,
            "face_id": obj.face_id,
            "timestamp": str(obj.last_seen),
            "metadata": metadata,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    self.method, self.url, json=payload, headers=self.headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    status = resp.status
                    body = await resp.text()
                    logger.debug(f"Webhook {self.url} response: {status}")
                    return {"url": self.url, "status": status, "body": body[:500]}
        except Exception as e:
            logger.warning(f"Webhook {self.url} failed: {e}")
            return {"url": self.url, "error": str(e)}
