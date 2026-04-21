"""APScheduler-based scheduler implementation (optional)."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal

import structlog
logger = structlog.get_logger()

from nanobot.optional.scheduler.service import CronService
from nanobot.optional.scheduler.types import CronJob, CronSchedule


class APSchedulerCronService(CronService):
    """
    Drop-in replacement for CronService that delegates scheduling to
    APScheduler's AsyncIOScheduler while reusing the parent's JSON
    store management and job execution logic.
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
    ):
        super().__init__(store_path=store_path, on_job=on_job)
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        self._scheduler = AsyncIOScheduler()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load jobs, register them with APScheduler, and start the scheduler."""
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._save_store()
        if self._store:
            for job in self._store.jobs:
                if job.enabled:
                    self._register_with_scheduler(job)
        self._scheduler.start()
        logger.info("APScheduler cron service started with {} jobs",
                    len(self._store.jobs) if self._store else 0)

    def stop(self) -> None:
        """Shutdown APScheduler and cancel any pending asyncio tasks."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Override timer arm – APScheduler drives its own wake-up loop
    # ------------------------------------------------------------------

    def _arm_timer(self) -> None:
        pass

    # ------------------------------------------------------------------
    # APScheduler helpers
    # ------------------------------------------------------------------

    def _register_with_scheduler(self, job: CronJob) -> None:
        """Register (or replace) a CronJob in APScheduler."""
        trigger = self._make_trigger(job.schedule)
        if trigger is None:
            logger.debug("APScheduler: skipping job {} – no valid trigger", job.id)
            return

        job_id = job.id  # captured by closure

        async def job_func() -> None:
            exec_job = self.get_job(job_id)
            if exec_job is None or not exec_job.enabled:
                return
            await self._execute_job(exec_job)
            # Sync next_run_at_ms from APScheduler's computed schedule (more accurate than estimate)
            ap_job = self._scheduler.get_job(job_id)
            if ap_job and ap_job.next_run_time:
                exec_job.state.next_run_at_ms = int(ap_job.next_run_time.timestamp() * 1000)
            self._save_store()
            # If _execute_job disabled the job (one-shot "at"), clean up from scheduler
            if not exec_job.enabled:
                try:
                    self._scheduler.remove_job(job_id)
                except Exception:
                    pass

        try:
            self._scheduler.add_job(
                job_func,
                trigger=trigger,
                id=job_id,
                name=job.name,
                replace_existing=True,
                misfire_grace_time=0,
                coalesce=False,
                max_instances=1,
            )
        except Exception as e:
            logger.warning("APScheduler: failed to register job {} ({}): {}", job_id, job.name, e)

    @staticmethod
    def _make_trigger(schedule: CronSchedule):
        """Convert a CronSchedule to an APScheduler trigger, or None if unmappable."""
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.date import DateTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        if schedule.kind == "cron" and schedule.expr:
            return CronTrigger.from_crontab(schedule.expr, timezone=schedule.tz or "UTC")

        if schedule.kind == "every" and schedule.every_ms and schedule.every_ms > 0:
            return IntervalTrigger(seconds=schedule.every_ms / 1000)

        if schedule.kind == "at" and schedule.at_ms:
            now_ms = int(time.time() * 1000)
            if schedule.at_ms <= now_ms:
                return None  # already in the past – parent will have disabled this job
            return DateTrigger(run_date=datetime.fromtimestamp(schedule.at_ms / 1000, tz=timezone.utc))

        return None

    # ------------------------------------------------------------------
    # Public API overrides – delegate to parent, then sync with APScheduler
    # ------------------------------------------------------------------

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        job = super().add_job(
            name=name, schedule=schedule, message=message,
            deliver=deliver, channel=channel, to=to,
            delete_after_run=delete_after_run,
        )
        if self._scheduler.running:
            self._register_with_scheduler(job)
        return job

    def register_system_job(self, job: CronJob) -> CronJob:
        result = super().register_system_job(job)
        if self._scheduler.running:
            self._register_with_scheduler(result)
        return result

    def remove_job(self, job_id: str) -> Literal["removed", "protected", "not_found"]:
        result = super().remove_job(job_id)
        if result == "removed":
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass
        return result

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        job = super().enable_job(job_id, enabled)
        if job and self._scheduler.running:
            if enabled:
                self._register_with_scheduler(job)
            else:
                try:
                    self._scheduler.remove_job(job_id)
                except Exception:
                    pass
        return job

    def status(self) -> dict:
        base = super().status()
        base["backend"] = "apscheduler"
        return base
