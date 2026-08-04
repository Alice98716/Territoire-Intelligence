"""Abstract base class shared by all specialized analysis agents."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import logging
import time

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Common contract and execution scaffolding for specialized agents.

    Subclasses (Demographer, Competitive, Regulatory, Real Estate, Economist,
    Synthesis, ...) implement `analyze` and `validate_inputs`. `execute` wraps
    both with input validation, timing, error handling, and a standardized
    response envelope, and should not be overridden.
    """

    def __init__(self, name: str, description: str) -> None:
        """Initializes the agent.

        Args:
            name: Short unique identifier for the agent (e.g. "demographer").
            description: Human-readable summary of what the agent analyzes.
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def analyze(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Performs the agent's core analysis.

        Args:
            intent: Structured request describing what to analyze (e.g.
                location, business type, and any upstream agent context).

        Returns:
            Agent-specific findings. This becomes the `data` field of the
            envelope returned by `execute`.

        Raises:
            Exception: Any failure is caught by `execute` and reported in
                the standardized error response.
        """
        raise NotImplementedError

    @abstractmethod
    async def validate_inputs(self, intent: dict[str, Any]) -> bool:
        """Checks whether `intent` has what `analyze` needs to run.

        Args:
            intent: Structured request to validate.

        Returns:
            True if `analyze` can run against `intent`, False otherwise.
        """
        raise NotImplementedError

    async def execute(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Runs the agent end-to-end: validate, analyze, time, report.

        This is the single entry point external callers (the orchestrator,
        the message queue consumer) should use. Do not override.

        Args:
            intent: Structured request passed through to `validate_inputs`
                and `analyze`.

        Returns:
            A standardized envelope:
                {
                    "agent": str,
                    "status": "success" | "invalid_input" | "error",
                    "data": Any | None,
                    "error": str | None,
                    "processing_time_ms": float,
                    "timestamp": str (ISO 8601, UTC),
                }
        """
        start = time.perf_counter()

        try:
            if not await self.validate_inputs(intent):
                logger.warning("%s: rejected invalid intent", self.name)
                return self._build_response(
                    status="invalid_input",
                    data=None,
                    error="Input validation failed",
                    start=start,
                )

            data = await self.analyze(intent)
            logger.info("%s: analysis completed", self.name)
            return self._build_response(status="success", data=data, error=None, start=start)

        except Exception as exc:  # noqa: BLE001 - boundary must not raise
            logger.exception("%s: analysis failed", self.name)
            return self._build_response(
                status="error",
                data=None,
                error=str(exc),
                start=start,
            )

    def _build_response(
        self,
        status: str,
        data: dict[str, Any] | None,
        error: str | None,
        start: float,
    ) -> dict[str, Any]:
        """Assembles the standardized response envelope.

        Args:
            status: One of "success", "invalid_input", "error".
            data: Agent output on success, None otherwise.
            error: Error description when status is not "success".
            start: `time.perf_counter()` value captured at the start of
                `execute`, used to compute elapsed processing time.

        Returns:
            The standardized envelope described in `execute`.
        """
        processing_time_ms = (time.perf_counter() - start) * 1000
        return {
            "agent": self.name,
            "status": status,
            "data": data,
            "error": error,
            "processing_time_ms": round(processing_time_ms, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
