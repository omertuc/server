"""Tests for a flow stream held for its player to ask for the rest after hanging up."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams import controller as controller_mod
from music_assistant.controllers.streams.controller import (
    FlowStreamConsumer,
    HeldFlowStream,
    StreamsController,
)

CHUNKS = [bytes([index]) * 1000 for index in range(10)]


class _Response:
    """A response that records what it sends, and fails once its player hangs up."""

    def __init__(self, hang_up_after: int | None = None) -> None:
        self.sent = bytearray()
        self.writes = 0
        self.hang_up_after = hang_up_after

    async def prepare(self, _request: object) -> None:
        """Prepare nothing."""

    async def write(self, data: bytes) -> None:
        """Send the data, unless the player hung up."""
        if self.hang_up_after is not None and self.writes >= self.hang_up_after:
            raise ConnectionResetError
        self.writes += 1
        self.sent += data


def _request(connected: bool = True) -> Any:
    """Return a request whose player is (still) connected or not."""
    return SimpleNamespace(transport=MagicMock() if connected else None)


def _controller() -> Any:
    """Return a streams controller streaming session-1 of queue-1."""
    controller: Any = StreamsController.__new__(StreamsController)
    controller.logger = MagicMock()
    controller._open_item_streams = {}
    controller._flow_streams = {}
    controller._active_output_streams = 0
    controller.mass = mass = MagicMock()
    mass.player_queues.queue_data.return_value = SimpleNamespace(session_id="session-1")
    mass.create_task.side_effect = asyncio.ensure_future
    return controller


def _stream(controller: Any, request: Any) -> tuple[HeldFlowStream, dict[str, bool]]:
    """Register a flow stream of CHUNKS for queue-1, and report whether it was closed."""
    state = {"closed": False}

    async def _audio() -> AsyncGenerator[bytes]:
        try:
            for chunk in CHUNKS:
                yield chunk
        finally:
            state["closed"] = True

    stream = HeldFlowStream(
        session_id="session-1",
        consumer=FlowStreamConsumer(request),
        audio_bytes=_audio(),
        headers={},
        http_profile="default",
        output_format=AudioFormat(content_type=ContentType.WAV),
    )
    controller._flow_streams["queue-1"] = stream
    return stream, state


async def _hang_up(controller: Any, stream: HeldFlowStream, after: int) -> _Response:
    """Send the stream until its player hangs up after the given number of chunks."""
    response = _Response(hang_up_after=after)
    controller._active_output_streams += 1
    assert await controller._send_flow_stream("queue-1", response, stream)
    request: Any = stream.consumer.request
    request.transport = None
    return response


@pytest.mark.asyncio
async def test_the_rest_continues_at_the_byte_the_player_asks_for() -> None:
    """What the player kept and what it gets after asking again is the stream, exactly."""
    controller = _controller()
    stream, state = _stream(controller, _request())
    first = await _hang_up(controller, stream, after=4)
    # the player kept only part of what reached it
    kept = bytes(first.sent[:3500])
    second = _Response()
    controller._flow_stream_response = MagicMock(return_value=second)
    request = _request()

    held = await controller._take_held_flow_stream("queue-1", "session-1", request, 3500)
    assert held is stream
    await controller._continue_flow_stream(request, "queue-1", held, 3500)

    assert kept + second.sent == b"".join(CHUNKS)
    assert state["closed"]
    assert controller._active_output_streams == 0


@pytest.mark.asyncio
async def test_a_held_stream_keeps_its_producer_going() -> None:
    """The producer behind it does not see its consumer gone while the stream is held."""
    controller = _controller()
    stream, state = _stream(controller, _request())

    await _hang_up(controller, stream, after=2)

    assert stream.consumer.connected()
    assert not state["closed"]
    await controller._close_flow_stream("queue-1", stream)


@pytest.mark.asyncio
async def test_a_held_stream_nobody_asks_for_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The player doesn't come back: the stream ends after a while."""
    monkeypatch.setattr(controller_mod, "FLOW_STREAM_HOLD_SECONDS", 0.01)
    controller = _controller()
    stream, state = _stream(controller, _request())

    await _hang_up(controller, stream, after=2)
    await asyncio.sleep(0.05)

    assert state["closed"]
    assert not stream.consumer.connected()
    assert controller._flow_streams == {}


@pytest.mark.asyncio
async def test_a_request_for_a_byte_no_longer_kept_starts_afresh() -> None:
    """Only what the stream sent last can be sent again."""
    controller = _controller()
    stream, _ = _stream(controller, _request())
    await _hang_up(controller, stream, after=4)

    assert await controller._take_held_flow_stream("queue-1", "session-1", _request(), 9999) is None
    await controller._close_flow_stream("queue-1", stream)


@pytest.mark.asyncio
async def test_a_second_request_next_to_a_connected_one_starts_afresh() -> None:
    """A request while the stream's player is still connected is not for the rest of it."""
    controller = _controller()
    stream, _ = _stream(controller, _request())

    assert await controller._take_held_flow_stream("queue-1", "session-1", _request(), 10) is None
    await controller._close_flow_stream("queue-1", stream)


@pytest.mark.asyncio
async def test_the_request_waits_for_the_stream_to_notice_its_player_hung_up() -> None:
    """The player asks again before the stream it left wrote again and found it gone."""
    controller = _controller()
    stream, _ = _stream(controller, _request(connected=False))

    async def _notice() -> None:
        await asyncio.sleep(0.1)
        controller._hold_flow_stream("queue-1", stream)

    notice = asyncio.create_task(_notice())
    request = _request()
    held = await controller._take_held_flow_stream("queue-1", "session-1", request, 0)
    await notice

    assert held is stream
    assert stream.consumer.request is request
    assert stream.expiry is None
    await controller._close_flow_stream("queue-1", stream)


@pytest.mark.asyncio
async def test_a_stream_of_a_session_that_ended_is_not_held() -> None:
    """After a session rotation the stream is over, whoever asks for the rest of it."""
    controller = _controller()
    stream, state = _stream(controller, _request())
    queue_data = controller.mass.player_queues.queue_data.return_value
    response = _Response(hang_up_after=2)
    write = response.write

    async def _write_then_rotate(data: bytes) -> None:
        await write(data)
        # a new session starts, and the sweep aborts the stale response
        queue_data.session_id = "session-2"

    response.write = _write_then_rotate  # type: ignore[method-assign]
    controller._active_output_streams += 1
    await controller._send_flow_stream("queue-1", response, stream)

    assert state["closed"]
    assert controller._flow_streams == {}
