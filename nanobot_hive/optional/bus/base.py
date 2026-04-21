"""Abstract message bus interface."""
from abc import ABC, abstractmethod


class MessageBusBase(ABC):
    """Decoupled async message bus."""

    @abstractmethod
    async def publish_inbound(self, message) -> None: ...

    @abstractmethod
    async def subscribe_inbound(self): ...

    @abstractmethod
    async def publish_outbound(self, message) -> None: ...

    @abstractmethod
    async def subscribe_outbound(self): ...
