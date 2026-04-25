"""Cron tool for scheduling reminders and tasks."""

from contextvars import ContextVar
from datetime import datetime
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from nanobot.telemetry import trace
from nanobot.cron import CronService
from nanobot.cron.types import CronJob, CronJobState, CronSchedule


class CronToolSchema(BaseModel):
    """Input schema for CronTool."""
    action: str = Field(description="Action to perform", enum=["add", "list", "remove"])
    name: str | None = Field( # This line had a syntax error.
        default=None,
        description="Optional short human-readable label for the job "
        "(e.g., 'weather-monitor', 'daily-standup'). Defaults to first 30 chars of message.",
    )
    message: str | None = Field(
        default=None,
        description="REQUIRED when action='add'. Instruction for the agent to execute when the job triggers "
        "(e.g., 'Send a reminder to WeChat: xxx' or 'Check system status and report'). "
        "Not used for action='list' or action='remove'."
    )
    every_seconds: int | None = Field(default=None, description="Interval in seconds (for recurring tasks)", ge=0)
    cron_expr: str | None = Field(default=None, description="Cron expression like '0 9 * * *' (for scheduled tasks)")
    tz: str | None = Field(
        default=None,
        description="Optional IANA timezone for cron expressions (e.g. 'America/Vancouver'). "
        "When omitted with cron_expr, the tool's default timezone applies."
    )
    at: str | None = Field(
        default=None,
        description="ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00'). "
        "Naive values use the tool's default timezone."
    )
    deliver: bool = Field(
        description="Whether to deliver the execution result to the user channel (default true)",
        default=True,
    )
    job_id: str | None = Field(default=None, description="REQUIRED when action='remove'. Job ID to remove (obtain via action='list').")

    @model_validator(mode="after")
    def validate_cron_parameters(self) -> "CronToolSchema":
        if self.action == "add":
            if not self.message:
                raise ValueError("message is required when action='add'")
            if not (self.every_seconds or self.cron_expr or self.at):
                raise ValueError("either every_seconds, cron_expr, or at is required when action='add'")
            if sum([bool(self.every_seconds), bool(self.cron_expr), bool(self.at)]) > 1:
                raise ValueError("only one of every_seconds, cron_expr, or at can be specified")
        elif self.action == "remove":
            if not self.job_id:
                raise ValueError("job_id is required when action='remove'")
        return self


class CronTool(BaseTool):
    """Tool to schedule reminders and recurring tasks."""
    name: str = "cron"
    description: str = (
        "Schedule reminders and recurring tasks. Actions: add, list, remove. "
        "If tz is omitted, cron expressions and naive ISO times default to UTC." # Default timezone will be set in __init__
    )
    args_schema: type[BaseModel] = CronToolSchema

    def __init__(self, cron_service: CronService, default_timezone: str = "UTC", **kwargs: Any):
        super().__init__(**kwargs)
        self._cron = cron_service
        self._default_timezone = default_timezone
        self._channel: ContextVar[str] = ContextVar("cron_channel", default="")
        self._chat_id: ContextVar[str] = ContextVar("cron_chat_id", default="")
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)
        # Update description with actual default timezone
        self.description = (
            "Schedule reminders and recurring tasks. Actions: add, list, remove. "
            f"If tz is omitted, cron expressions and naive ISO times default to {self._default_timezone}."
        )

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current session context for delivery."""
        self._channel.set(channel)
        self._chat_id.set(chat_id)

    def set_cron_context(self, active: bool):
        """Mark whether the tool is executing inside a cron job callback."""
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token) -> None:
        """Restore previous cron context."""
        self._in_cron_context.reset(token)

    @staticmethod
    def _validate_timezone(tz: str) -> str | None:
        from zoneinfo import ZoneInfo

        try:
            ZoneInfo(tz)
        except (KeyError, Exception):
            return f"Error: unknown timezone '{tz}'"
        return None

    def _display_timezone(self, schedule: CronSchedule) -> str:
        """Pick the most human-meaningful timezone for display."""
        return schedule.tz or self._default_timezone

    @staticmethod
    def _format_timestamp(ms: int, tz_name: str) -> str:
        from zoneinfo import ZoneInfo

        dt = datetime.fromtimestamp(ms / 1000, tz=ZoneInfo(tz_name))
        return f"{dt.isoformat()} ({tz_name})"

    async def _arun(
        self,
        action: str,
        name: str | None = None,
        message: str | None = None,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        deliver: bool = True,
    ) -> str:
        # The validation is now handled by Pydantic's model_validator in CronToolSchema
        # So we can remove the manual validate_params call.
        # Also, message is now Optional in _arun, but required by schema for 'add' action.
        # We need to ensure message is not None if action is 'add'.
        if action == "add" and message is None:
            # This case should be caught by the model_validator, but as a safeguard
            return "Error: message is required when action='add'"

        async with trace(f"cron.{action}"):
            if action == "add":
                if self._in_cron_context.get():
                    return "Error: cannot schedule new jobs from within a cron job execution"
                return self._add_job(name, message, every_seconds, cron_expr, tz, at, deliver)
            elif action == "list":
                return self._list_jobs()
            elif action == "remove":
                return self._remove_job(job_id)
            return f"Unknown action: {action}"

    def _add_job(
        self,
        name: str | None,
        message: str | None, # message can be None here due to schema, but model_validator ensures it's not for 'add'
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
        deliver: bool = True,
    ) -> str:
        # message is guaranteed to be not None here if action was 'add' due to model_validator
        if message is None: # Should not happen due to model_validator
            return "Internal Error: message should not be None for 'add' action."

        channel = self._channel.get()
        chat_id = self._chat_id.get()
        if not channel or not chat_id:
            return "Error: no session context (channel/chat_id)"
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"
        if tz:
            if err := self._validate_timezone(tz):
                return err

        # Build schedule
        delete_after = False
        if every_seconds is not None: # Check for None explicitly
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            effective_tz = tz or self._default_timezone
            if err := self._validate_timezone(effective_tz):
                return err
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=effective_tz)
        elif at:
            from zoneinfo import ZoneInfo

            try:
                dt = datetime.fromisoformat(at)
            except ValueError:
                return f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS"
            if dt.tzinfo is None:
                if err := self._validate_timezone(self._default_timezone):
                    return err
                dt = dt.replace(tzinfo=ZoneInfo(self._default_timezone))
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            # This case should be caught by the model_validator, but as a safeguard
            return "Error: either every_seconds, cron_expr, or at is required"

        job = self._cron.add_job(
            name=name or message[:30],
            schedule=schedule,
            message=message,
            deliver=deliver,
            channel=channel,
            to=chat_id,
            delete_after_run=delete_after,
        )
        return f"Created job '{job.name}' (id: {job.id})"

    def _format_timing(self, schedule: CronSchedule) -> str:
        """Format schedule as a human-readable timing string."""
        if schedule.kind == "cron":
            tz = f" ({schedule.tz})" if schedule.tz else ""
            return f"cron: {schedule.expr}{tz}"
        if schedule.kind == "every" and schedule.every_ms:
            ms = schedule.every_ms
            if ms % 3_600_000 == 0:
                return f"every {ms // 3_600_000}h"
            if ms % 60_000 == 0:
                return f"every {ms // 60_000}m"
            if ms % 1000 == 0:
                return f"every {ms // 1000}s"
            return f"every {ms}ms"
        if schedule.kind == "at" and schedule.at_ms:
            return f"at {self._format_timestamp(schedule.at_ms, self._display_timezone(schedule))}"
        return schedule.kind

    def _format_state(self, state: CronJobState, schedule: CronSchedule) -> list[str]:
        """Format job run state as display lines."""
        lines: list[str] = []
        display_tz = self._display_timezone(schedule)
        if state.last_run_at_ms:
            info = (
                f"  Last run: {self._format_timestamp(state.last_run_at_ms, display_tz)}"
                f" — {state.last_status or 'unknown'}"
            )
            if state.last_error:
                info += f" ({state.last_error})"
            lines.append(info)
        if state.next_run_at_ms:
            lines.append(f"  Next run: {self._format_timestamp(state.next_run_at_ms, display_tz)}")
        return lines

    @staticmethod
    def _system_job_purpose(job: CronJob) -> str:
        if job.name == "dream":
            return "Dream memory consolidation for long-term memory."
        return "System-managed internal job."

    def _list_jobs(self) -> str:
        jobs = self._cron.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines = []
        for j in jobs:
            timing = self._format_timing(j.schedule)
            parts = [f"- {j.name} (id: {j.id}, {timing})"]
            if j.payload.kind == "system_event":
                parts.append(f"  Purpose: {self._system_job_purpose(j)}")
                parts.append("  Protected: visible for inspection, but cannot be removed.")
            parts.extend(self._format_state(j.state, j.schedule))
            lines.append("\n".join(parts))
        return "Scheduled jobs:\n" + "\n".join(lines)

    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        result = self._cron.remove_job(job_id)
        if result == "removed":
            return f"Removed job {job_id}"
        if result == "protected":
            job = self._cron.get_job(job_id)
            if job and job.name == "dream":
                return (
                    "Cannot remove job `dream`.\n"
                    "This is a system-managed Dream memory consolidation job for long-term memory.\n"
                    "It remains visible so you can inspect it, but it cannot be removed."
                )
            return (
                f"Cannot remove job `{job_id}`.\n"
                "This is a protected system-managed cron job."
            )
        return f"Job {job_id} not found"
        "Action-specific parameters: add requires a non-empty message plus one schedule "
