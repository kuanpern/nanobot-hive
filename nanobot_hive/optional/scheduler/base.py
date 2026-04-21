"""Abstract scheduler interface."""
from abc import ABC, abstractmethod


class SchedulerBase(ABC):
    """Cron-like scheduling interface."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def add_job(self, job) -> None: ...

    @abstractmethod
    def remove_job(self, job_id: str) -> None: ...

    @abstractmethod
    def list_jobs(self) -> list: ...
