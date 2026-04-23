from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import structlog

logger = structlog.get_logger()

class BaseChannel(ABC):
    """Abstract base class for all channel implementations."""
    
    def __init__(self, config: Any, bus: Any):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, msg: Any) -> None: ...

    @property
    def is_running(self) -> bool:
        return self._running