# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
from typing import Any

from loguru import logger

from .base import BaseAction
from .webhook import WebhookAction
from .mqtt import MQTTAction
from .logger_action import LogAction
from ..storage.models import TrackedObject


class ActionDispatcher:
    """Central dispatcher: matches triggers and runs configured actions."""

    def __init__(self):
        self._triggers: list[dict] = []
        self._action_cache: dict[str, list[BaseAction]] = {}
        self._fired: set[tuple[str, str]] = set()
        self._fired_names: set[tuple[str, str]] = set()

    def load_triggers(self, triggers_config: list[dict]):
        """Load triggers from parsed YAML config."""
        self._triggers = triggers_config or []
        self._action_cache.clear()
        self._fired.clear()
        self._fired_names.clear()

        for trigger in self._triggers:
            actions = []
            for act in trigger.get("actions", []):
                a_type = act.get("type")
                if a_type == "webhook":
                    actions.append(WebhookAction(
                        url=act["url"],
                        method=act.get("method", "POST"),
                        headers=act.get("headers"),
                    ))
                elif a_type == "mqtt":
                    actions.append(MQTTAction(
                        topic=act["topic"],
                        qos=act.get("qos", 1),
                    ))
                elif a_type == "log":
                    actions.append(LogAction(level=act.get("level", "INFO")))
            self._action_cache[trigger["name"]] = actions

        logger.info(f"Loaded {len(self._triggers)} triggers")

    async def evaluate(
        self,
        obj: TrackedObject,
        plate_number: str | None,
        face_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Check all triggers against object, run matching actions. Returns list of results."""
        meta = metadata or {}
        results = []

        for trigger in self._triggers:
            t_type = trigger.get("type")
            matched = False
            match_value = None

            if t_type == "plate":
                if plate_number and plate_number in trigger.get("values", []):
                    matched = True
                    match_value = plate_number

            elif t_type == "class":
                if obj.class_name in trigger.get("values", []):
                    matched = True
                    match_value = obj.class_name

            elif t_type == "face":
                if face_id:
                    if trigger.get("source_db", False):
                        matched = True
                        match_value = face_id
                    elif face_id in trigger.get("values", []):
                        matched = True
                        match_value = face_id

            if matched:
                once_per_object = trigger.get("once_per_object", True)
                fired_key = (trigger["name"], str(obj.id))
                is_departure = meta.get("event") in ("departed", "reappeared")
                if once_per_object and not is_departure and fired_key in self._fired:
                    continue
                # Name-based dedup: same trigger + same name = one object across cameras
                if once_per_object and not is_departure and obj.name:
                    name_key = (trigger["name"], obj.name)
                    if name_key in self._fired_names:
                        continue
                meta["match_value"] = match_value
                for action in self._action_cache.get(trigger["name"], []):
                    try:
                        result = await action.execute(obj, trigger["name"], meta)
                        results.append({
                            "trigger": trigger["name"],
                            "action_type": action.__class__.__name__,
                            "result": result,
                        })
                        if once_per_object and not is_departure:
                            self._fired.add(fired_key)
                            if obj.name:
                                self._fired_names.add((trigger["name"], obj.name))
                    except Exception as e:
                        logger.error(f"Action {trigger['name']} failed: {e}")
                        results.append({
                            "trigger": trigger["name"],
                            "action_type": action.__class__.__name__,
                            "error": str(e),
                        })

        return results
