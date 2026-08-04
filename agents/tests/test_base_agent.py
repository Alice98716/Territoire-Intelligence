"""Unit tests for agents.base_agent.BaseAgent."""

import asyncio

import pytest

from agents.base_agent import BaseAgent


class ConcreteAgent(BaseAgent):
    """Minimal working subclass used to exercise execute()."""

    def __init__(self, result: dict | None = None, should_raise: bool = False, delay: float = 0.0):
        super().__init__(name="concrete", description="test agent")
        self._result = result if result is not None else {"ok": True}
        self._should_raise = should_raise
        self._delay = delay

    async def validate_inputs(self, intent: dict) -> bool:
        return "location" in intent

    async def analyze(self, intent: dict) -> dict:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._should_raise:
            raise ValueError("analysis blew up")
        return self._result


def run(coro):
    return asyncio.run(coro)


def test_cannot_instantiate_base_agent_directly():
    with pytest.raises(TypeError):
        BaseAgent(name="base", description="abstract")


def test_subclass_must_implement_both_abstract_methods():
    class OnlyAnalyze(BaseAgent):
        async def analyze(self, intent: dict) -> dict:
            return {}

    class OnlyValidate(BaseAgent):
        async def validate_inputs(self, intent: dict) -> bool:
            return True

    with pytest.raises(TypeError):
        OnlyAnalyze(name="only-analyze", description="partial")
    with pytest.raises(TypeError):
        OnlyValidate(name="only-validate", description="partial")


def test_execute_success_returns_standardized_envelope():
    agent = ConcreteAgent(result={"population": 12345})
    response = run(agent.execute({"location": "Montreal"}))

    assert response["agent"] == "concrete"
    assert response["status"] == "success"
    assert response["data"] == {"population": 12345}
    assert response["error"] is None
    assert "processing_time_ms" in response
    assert "timestamp" in response


def test_execute_invalid_input_short_circuits_analyze():
    agent = ConcreteAgent()
    response = run(agent.execute({}))  # missing required "location" key

    assert response["status"] == "invalid_input"
    assert response["data"] is None
    assert response["error"] is not None


def test_execute_handles_exceptions_from_analyze():
    agent = ConcreteAgent(should_raise=True)
    response = run(agent.execute({"location": "Montreal"}))

    assert response["status"] == "error"
    assert response["data"] is None
    assert "analysis blew up" in response["error"]


def test_execute_captures_processing_time():
    agent = ConcreteAgent(delay=0.05)
    response = run(agent.execute({"location": "Montreal"}))

    assert isinstance(response["processing_time_ms"], float)
    assert response["processing_time_ms"] >= 50
