"""Tests for a player that reconnects to its flow stream continuing where it got to."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, CrossfadeMode, MediaType
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player_queue import PlayLogEntry

from music_assistant.controllers.streams.audio import StreamsAudio
from music_assistant.controllers.streams.controller import StreamsController

PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE, sample_rate=44100, bit_depth=16, channels=2
)
QUEUE = cast(
    "Any",
    SimpleNamespace(
        queue_id="queue-1",
        display_name="Queue",
        flow_mode=False,
        overlay_enabled=False,
        overlay_source=None,
    ),
)


def _queue_item(item_id: str, seek_position: int = 0) -> Any:
    """Return a queue item a flow stream can play."""
    return SimpleNamespace(
        queue_item_id=item_id,
        name=item_id,
        media_type=MediaType.TRACK,
        duration=300,
        extra_attributes={},
        streamdetails=SimpleNamespace(
            fade_in=False,
            stream_error=False,
            tail_overlap=None,
            uri=f"test://{item_id}",
            seek_position=seek_position,
            duration=300,
            buffer=None,
            seconds_streamed=None,
            is_realtime=False,
            audio_format=PCM_FORMAT,
        ),
    )


def _audio(queue_data: SimpleNamespace) -> StreamsAudio:
    """Return a flow stream producer whose items each stream 60 seconds of silence."""
    mass = MagicMock()
    mass.player_queues.queue_data.return_value = queue_data
    mass.streams.get_crossfade_mode.return_value = CrossfadeMode.DISABLED
    mass.config.get_raw_core_config_value.return_value = 0
    mass.player_queues.get_active_queue.return_value = None

    async def _load_next(*_args: object) -> Any:
        raise QueueEmpty

    mass.player_queues.load_next_queue_item = _load_next
    audio = StreamsAudio(cast("Any", mass))

    async def _item_stream(*_args: object, **_kwargs: object) -> AsyncGenerator[bytes]:
        for _ in range(60):
            yield b"\x00" * PCM_FORMAT.pcm_sample_size

    audio.get_queue_item_stream = _item_stream  # type: ignore[method-assign]
    return audio


def _controller(flow_log: list[PlayLogEntry], *items: Any) -> Any:
    """Return a streams controller for a queue with the given play log and items."""
    controller: Any = StreamsController.__new__(StreamsController)
    controller.logger = logging.getLogger(__name__)
    controller._open_item_streams = {}
    controller.mass = mass = MagicMock()
    mass.player_queues.queue_data.return_value = SimpleNamespace(flow_mode_stream_log=flow_log)
    by_id = {item.queue_item_id: item for item in items}
    mass.player_queues.get_item.side_effect = lambda _queue_id, item_id: by_id.get(item_id)
    return controller


def _open_request(controller: Any, *, connected: bool) -> tuple[str, Any]:
    """Register an open flow stream request of session-1 on the controller."""
    entry = ("session-1", SimpleNamespace(transport=MagicMock() if connected else None))
    controller._open_item_streams.setdefault("queue-1", []).append(entry)
    return entry


@pytest.mark.asyncio
async def test_a_stream_closed_halfway_through_an_item_records_how_far_it_got() -> None:
    """The player hangs up mid-item: its play log entry gets the seconds that went out."""
    queue_data = SimpleNamespace(session_id="session-1", flow_mode_stream_log=[])
    stream = _audio(queue_data).get_queue_flow_stream(
        QUEUE, _queue_item("item-1"), PCM_FORMAT, session_id="session-1"
    )
    for _ in range(3):
        await anext(stream)

    await stream.aclose()

    # the slice it was suspended on when closed is not counted
    assert queue_data.flow_mode_stream_log[0].seconds_streamed == 2.0


@pytest.mark.asyncio
async def test_a_reconnected_stream_carries_on_the_play_log() -> None:
    """The earlier streams' entries stay in front of the new stream's own."""
    earlier = [PlayLogEntry("item-0", seconds_streamed=200), PlayLogEntry("item-1", 50, 50, 0)]
    queue_data = SimpleNamespace(session_id="session-1", flow_mode_stream_log=earlier)
    stream = _audio(queue_data).get_queue_flow_stream(
        QUEUE,
        _queue_item("item-1", seek_position=50),
        PCM_FORMAT,
        session_id="session-1",
        play_log=earlier,
    )
    await anext(stream)

    log = queue_data.flow_mode_stream_log
    assert log is not earlier
    assert log[:2] == earlier
    assert (log[2].queue_item_id, log[2].seek_position) == ("item-1", 50)
    await stream.aclose()


@pytest.mark.asyncio
async def test_a_reconnect_continues_the_item_where_the_previous_stream_got_to() -> None:
    """The item the previous stream was on is seeked past what it already sent."""
    item = _queue_item("item-1")
    flow_log = [
        PlayLogEntry("item-0", seconds_streamed=200),
        PlayLogEntry("item-1", seconds_streamed=52.5, seek_position=30),
    ]
    controller = _controller(flow_log, item)

    start_item, play_log = await controller._get_flow_start_point(QUEUE, "session-1", "item-0")

    assert start_item is item
    assert item.streamdetails.seek_position == 82
    assert play_log == flow_log


@pytest.mark.asyncio
async def test_a_reconnect_after_a_probe_starts_the_item_over() -> None:
    """A previous stream that only just started the item leaves no entry; the item restarts."""
    item = _queue_item("item-1")
    flow_log = [
        PlayLogEntry("item-0", seconds_streamed=200),
        PlayLogEntry("item-1", seconds_streamed=3, seek_position=0),
    ]
    controller = _controller(flow_log, item)

    start_item, play_log = await controller._get_flow_start_point(QUEUE, "session-1", "item-0")

    assert start_item is item
    assert item.streamdetails.seek_position == 0
    assert play_log == flow_log[:1]


@pytest.mark.asyncio
async def test_a_second_stream_next_to_a_connected_one_is_no_reconnect() -> None:
    """While the first stream is still connected, the second starts as before."""
    current = _queue_item("item-0")
    flow_log = [PlayLogEntry("item-1", seconds_streamed=None)]
    controller = _controller(flow_log, current, _queue_item("item-1"))
    controller._get_flow_start_item = MagicMock(return_value=current)
    _open_request(controller, connected=True)

    start_item, play_log = await controller._get_flow_start_point(QUEUE, "session-1", "item-0")

    assert start_item is current
    assert play_log is None


@pytest.mark.asyncio
async def test_a_reconnect_waits_for_the_previous_stream_to_record_how_far_it_got() -> None:
    """The player asks again before the stream it left has finished tearing down."""
    item = _queue_item("item-1")
    flow_log = [PlayLogEntry("item-1", seconds_streamed=None, seek_position=0)]
    controller = _controller(flow_log, item)
    entry = _open_request(controller, connected=False)

    async def _tear_down() -> None:
        await asyncio.sleep(0.1)
        flow_log[0].seconds_streamed = 40
        controller._open_item_streams["queue-1"].remove(entry)

    tear_down = asyncio.create_task(_tear_down())
    start_item, play_log = await controller._get_flow_start_point(QUEUE, "session-1", "item-1")
    await tear_down

    assert start_item is item
    assert item.streamdetails.seek_position == 40
    assert play_log == flow_log


@pytest.mark.asyncio
async def test_a_fresh_session_starts_at_its_first_item() -> None:
    """Without an earlier stream in the session there is nothing to continue."""
    first = _queue_item("item-0")
    controller = _controller([], first)
    controller._get_flow_start_item = MagicMock(return_value=first)

    start_item, play_log = await controller._get_flow_start_point(QUEUE, "session-1", "item-0")

    assert start_item is first
    assert play_log is None
    assert first.streamdetails.seek_position == 0


@pytest.mark.asyncio
async def test_a_second_reconnect_in_the_same_item_continues_past_the_first() -> None:
    """An entry without its own start position began at the item's latest seek."""
    item = _queue_item("item-1", seek_position=50)
    flow_log = [
        PlayLogEntry("item-1", seconds_streamed=50, seek_position=0),
        PlayLogEntry("item-1", seconds_streamed=40),
    ]
    controller = _controller(flow_log, item)

    start_item, play_log = await controller._get_flow_start_point(QUEUE, "session-1", "item-1")

    assert start_item is item
    assert item.streamdetails.seek_position == 90
    assert play_log == flow_log
