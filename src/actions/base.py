# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
from abc import ABC, abstractmethod
from typing import Any

from ..storage.models import TrackedObject


class BaseAction(ABC):
    """Abstract trigger action."""

    @abstractmethod
    async def execute(self, obj: TrackedObject, trigger_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Execute action, returns result dict."""
        ...
