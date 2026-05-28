import sys
from typing import Any

from loguru import logger

from .base import BaseAction
from ..storage.models import TrackedObject


class LogAction(BaseAction):
    """Log trigger event via loguru."""

    def __init__(self, level: str = "INFO"):
        self.level = level.upper()

    async def execute(self, obj: TrackedObject, trigger_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
        msg = (
            f"[TRIGGER:{trigger_name}] "
            f"object={obj.id} class={obj.class_name} "
            f"camera={obj.camera_id} "
            f"plate={obj.plate_number or '-'} "
            f"face={obj.face_id or '-'}"
        )
        logger.log(self.level, msg)
        return {"logged": True}
