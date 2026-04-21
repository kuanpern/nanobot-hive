"""Abstract web tool interface."""
from abc import ABC, abstractmethod


class WebFetchBase(ABC):
    @abstractmethod
    async def fetch(self, url: str, **kwargs) -> str: ...


class WebSearchBase(ABC):
    @abstractmethod
    async def search(self, query: str, **kwargs) -> list: ...
