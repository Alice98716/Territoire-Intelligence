"""Redis pub/sub abstraction for inter-agent messaging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import json
import logging

import redis.asyncio as redis
from redis.asyncio.client import PubSub

logger = logging.getLogger(__name__)

DIRECT_CHANNEL_PREFIX = "agent:"


class MessageQueueError(Exception):
    """Raised when a publish/subscribe/send operation cannot complete."""


class MessageQueue:
    """Thin async wrapper around Redis pub/sub for agent-to-agent messages.

    Holds a single pooled `redis.asyncio.Redis` client (Redis's own
    connection pool handles reuse), and standardizes message envelopes as:
        {
            "timestamp": str (ISO 8601, UTC),
            "sender_agent": str,
            "receiver_agent": str | None,
            "payload": Any,
        }
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", **pool_kwargs: Any) -> None:
        """Initializes the queue with a pooled Redis client.

        Args:
            redis_url: Redis connection string, e.g.
                "redis://localhost:6379/0".
            **pool_kwargs: Extra keyword arguments forwarded to
                `redis.asyncio.from_url` (e.g. `max_connections`).
        """
        self._redis_url = redis_url
        self._client: redis.Redis = redis.from_url(
            redis_url, decode_responses=True, **pool_kwargs
        )

    async def close(self) -> None:
        """Closes the underlying connection pool."""
        await self._client.aclose()

    async def publish(self, channel: str, message: dict[str, Any]) -> int:
        """Publishes a message to `channel`.

        Args:
            channel: Topic name, typically the receiving agent's name or a
                broadcast topic such as "agent:results".
            message: Arbitrary JSON-serializable payload. Wrapped in the
                standard envelope; the "payload" key is set to this value.

        Returns:
            Number of subscribers that received the message.

        Raises:
            MessageQueueError: If the message cannot be published.
        """
        envelope = self._build_envelope(
            sender_agent=message.get("sender_agent", "unknown"),
            receiver_agent=None,
            payload=message,
        )
        try:
            return await self._client.publish(channel, json.dumps(envelope))
        except (redis.RedisError, TypeError) as exc:
            logger.exception("Failed to publish to channel %r", channel)
            raise MessageQueueError(f"publish to {channel!r} failed") from exc

    async def subscribe(self, channel: str) -> AsyncIterator[dict[str, Any]]:
        """Subscribes to `channel` and yields decoded messages as they arrive.

        Intended usage:
            async for message in queue.subscribe("agent:demographer"):
                handle(message)

        Args:
            channel: Topic name to listen on.

        Yields:
            Decoded message envelopes published on `channel`.

        Raises:
            MessageQueueError: If the subscription cannot be established.
        """
        pubsub: PubSub = self._client.pubsub()
        try:
            await pubsub.subscribe(channel)
        except redis.RedisError as exc:
            logger.exception("Failed to subscribe to channel %r", channel)
            raise MessageQueueError(f"subscribe to {channel!r} failed") from exc

        try:
            async for raw in pubsub.listen():
                if raw.get("type") != "message":
                    continue
                try:
                    yield json.loads(raw["data"])
                except json.JSONDecodeError:
                    logger.warning("Dropped malformed message on channel %r", channel)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def send_direct(
        self, agent_id: str, message: dict[str, Any], sender_agent: Optional[str] = None
    ) -> int:
        """Sends a message directly to a single agent's private channel.

        Args:
            agent_id: Name of the receiving agent. Delivered on the channel
                f"{DIRECT_CHANNEL_PREFIX}{agent_id}".
            message: Arbitrary JSON-serializable payload.
            sender_agent: Name of the sending agent, recorded in the
                envelope. Defaults to message.get("sender_agent", "unknown").

        Returns:
            Number of subscribers that received the message (0 or 1 for a
            direct channel, unless multiple listeners share it).

        Raises:
            MessageQueueError: If the message cannot be sent.
        """
        channel = f"{DIRECT_CHANNEL_PREFIX}{agent_id}"
        envelope = self._build_envelope(
            sender_agent=sender_agent or message.get("sender_agent", "unknown"),
            receiver_agent=agent_id,
            payload=message,
        )
        try:
            return await self._client.publish(channel, json.dumps(envelope))
        except (redis.RedisError, TypeError) as exc:
            logger.exception("Failed to send direct message to %r", agent_id)
            raise MessageQueueError(f"send_direct to {agent_id!r} failed") from exc

    @staticmethod
    def _build_envelope(
        sender_agent: str, receiver_agent: Optional[str], payload: Any
    ) -> dict[str, Any]:
        """Builds the standard message envelope."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender_agent": sender_agent,
            "receiver_agent": receiver_agent,
            "payload": payload,
        }
