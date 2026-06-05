import asyncio
import json
import os
import socket
import time

import pytest
import websockets
from xprocess import ProcessStarter


PROJECT_DIR = "/home/user/myproject"
WS_URL = "ws://127.0.0.1:8765"


@pytest.fixture(scope="session")
def start_server(xprocess):
    class Starter(ProcessStarter):
        name = "ws_search_server"
        args = ["python3", "server.py"]
        env = os.environ.copy()
        popen_kwargs = {
            "cwd": PROJECT_DIR,
            "text": True,
        }
        timeout = 60
        terminate_on_interrupt = True

        def startup_check(self):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                return s.connect_ex(("127.0.0.1", 8765)) == 0

    xprocess.ensure(Starter.name, Starter)
    yield
    info = xprocess.getinfo(Starter.name)
    info.terminate()


async def _send_and_collect(payload, timeout=10.0, expected_frames=None):
    """Open a fresh connection, send one message, collect frames until done or timeout."""
    async with websockets.connect(WS_URL, open_timeout=10, close_timeout=2) as ws:
        await ws.send(json.dumps(payload))
        frames = []
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=end - time.monotonic())
            except asyncio.TimeoutError:
                break
            except websockets.ConnectionClosed:
                break
            obj = json.loads(raw)
            frames.append(obj)
            if isinstance(obj, dict) and obj.get("done") is True:
                break
            if expected_frames is not None and len(frames) >= expected_frames:
                break
        return frames


def _split_frames(frames):
    """Return (hit_frames, done_frame_or_None)."""
    hits = [f for f in frames if "done" not in f]
    done = next((f for f in frames if isinstance(f, dict) and f.get("done") is True), None)
    return hits, done


def _assert_hit_schema(frame):
    assert isinstance(frame, dict), f"Per-hit frame must be a JSON object, got {type(frame)}"
    assert set(frame.keys()) == {"rank", "id", "score", "text"}, (
        f"Per-hit frame must have exactly keys {{rank,id,score,text}}; got {set(frame.keys())}"
    )
    assert isinstance(frame["rank"], int) and not isinstance(frame["rank"], bool), "rank must be int"
    assert isinstance(frame["id"], int) and not isinstance(frame["id"], bool), "id must be int"
    assert isinstance(frame["score"], (int, float)) and not isinstance(frame["score"], bool), "score must be a number"
    assert isinstance(frame["text"], str) and len(frame["text"]) > 0, "text must be a non-empty string"


def _assert_done_schema(frame, expected_total=None):
    assert isinstance(frame, dict), "Done frame must be a dict"
    assert set(frame.keys()) == {"done", "total", "elapsed_ms"}, (
        f"Done frame must have exactly keys {{done,total,elapsed_ms}}; got {set(frame.keys())}"
    )
    assert frame["done"] is True, "done must be true"
    assert isinstance(frame["total"], int) and not isinstance(frame["total"], bool) and frame["total"] >= 0, (
        "total must be non-negative int"
    )
    assert isinstance(frame["elapsed_ms"], (int, float)) and not isinstance(frame["elapsed_ms"], bool) and frame["elapsed_ms"] >= 0, (
        "elapsed_ms must be a non-negative number"
    )
    if expected_total is not None:
        assert frame["total"] == expected_total, (
            f"done.total expected {expected_total}, got {frame['total']}"
        )


def test_fts_anchor_quantumtoken42(start_server):
    frames = asyncio.run(
        _send_and_collect({"query": "QUANTUMTOKEN42UNIQUE", "k": 5, "mode": "fts"})
    )
    hits, done = _split_frames(frames)
    assert len(hits) == 5, f"Expected 5 per-hit frames, got {len(hits)}: {hits}"
    for i, h in enumerate(hits, start=1):
        _assert_hit_schema(h)
        assert h["rank"] == i, f"Per-hit frames must be in rank order 1..k; got rank {h['rank']} at position {i}"
    assert hits[0]["id"] == 42, f"FTS rank-1 for QUANTUMTOKEN42UNIQUE must be id 42, got {hits[0]['id']}"
    scores = [h["score"] for h in hits]
    assert all(scores[i] >= scores[i + 1] - 1e-6 for i in range(len(scores) - 1)), (
        f"FTS scores must be monotonically non-increasing (BM25), got {scores}"
    )
    assert done is not None, "Expected a final done frame"
    _assert_done_schema(done, expected_total=5)


def test_fts_anchor_vortextoken99(start_server):
    frames = asyncio.run(
        _send_and_collect({"query": "VORTEXTOKEN99UNIQUE", "k": 3, "mode": "fts"})
    )
    hits, done = _split_frames(frames)
    assert len(hits) == 3, f"Expected 3 per-hit frames, got {len(hits)}"
    for h in hits:
        _assert_hit_schema(h)
    assert hits[0]["id"] == 99, f"FTS rank-1 for VORTEXTOKEN99UNIQUE must be id 99, got {hits[0]['id']}"
    _assert_done_schema(done, expected_total=3)


def test_vector_mode_anchor42(start_server):
    frames = asyncio.run(
        _send_and_collect({"query": "QUANTUMTOKEN42UNIQUE", "k": 5, "mode": "vector"})
    )
    hits, done = _split_frames(frames)
    assert len(hits) == 5, f"Expected 5 per-hit frames, got {len(hits)}"
    for i, h in enumerate(hits, start=1):
        _assert_hit_schema(h)
        assert h["rank"] == i, f"Per-hit frames must be in rank order; got rank {h['rank']} at pos {i}"
    assert hits[0]["id"] == 42, (
        f"Vector rank-1 for embed_text('QUANTUMTOKEN42UNIQUE') must be id 42 (rigged vector); got {hits[0]['id']}"
    )
    scores = [h["score"] for h in hits]
    assert all(scores[i] <= scores[i + 1] + 1e-6 for i in range(len(scores) - 1)), (
        f"Vector mode scores (distances) must be non-decreasing, got {scores}"
    )
    _assert_done_schema(done, expected_total=5)


def test_hybrid_mode_quantumtoken42(start_server):
    frames = asyncio.run(
        _send_and_collect({"query": "QUANTUMTOKEN42UNIQUE", "k": 5, "mode": "hybrid"})
    )
    hits, done = _split_frames(frames)
    assert len(hits) == 5, f"Expected 5 per-hit frames, got {len(hits)}"
    for h in hits:
        _assert_hit_schema(h)
    assert hits[0]["id"] == 42, (
        f"Hybrid rank-1 for QUANTUMTOKEN42UNIQUE must be id 42, got {hits[0]['id']}"
    )
    scores = [h["score"] for h in hits]
    assert all(scores[i] >= scores[i + 1] - 1e-6 for i in range(len(scores) - 1)), (
        f"Hybrid relevance scores must be monotonically non-increasing, got {scores}"
    )
    _assert_done_schema(done, expected_total=5)


async def _debounce_scenario():
    """One connection: send, drain, immediately send duplicate, expect silence; wait, send again, expect full stream."""
    async with websockets.connect(WS_URL, open_timeout=10, close_timeout=2) as ws:
        payload = json.dumps({"query": "QUANTUMTOKEN42UNIQUE", "k": 4, "mode": "fts"})

        # First message: collect 4 hit frames + 1 done frame.
        await ws.send(payload)
        first_frames = []
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            raw = await asyncio.wait_for(ws.recv(), timeout=end - time.monotonic())
            obj = json.loads(raw)
            first_frames.append(obj)
            if isinstance(obj, dict) and obj.get("done") is True:
                break

        # Immediately send duplicate (within debounce window).
        t_dup_sent = time.monotonic()
        await ws.send(payload)

        # Listen for up to 250 ms; expect no frames.
        dup_frames = []
        try:
            while time.monotonic() - t_dup_sent < 0.25:
                remaining = 0.25 - (time.monotonic() - t_dup_sent)
                if remaining <= 0:
                    break
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                dup_frames.append(json.loads(raw))
        except asyncio.TimeoutError:
            pass

        # Sleep beyond the 100 ms window to ensure the next identical message is accepted.
        await asyncio.sleep(0.20)

        # Third send: expect full stream again.
        await ws.send(payload)
        third_frames = []
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            raw = await asyncio.wait_for(ws.recv(), timeout=end - time.monotonic())
            obj = json.loads(raw)
            third_frames.append(obj)
            if isinstance(obj, dict) and obj.get("done") is True:
                break

        return first_frames, dup_frames, third_frames


def test_debounce_duplicate_within_window(start_server):
    first, dup, third = asyncio.run(_debounce_scenario())

    first_hits, first_done = _split_frames(first)
    assert len(first_hits) == 4, f"Initial request must stream 4 hit frames, got {len(first_hits)}"
    assert first_done is not None and first_done.get("done") is True, "Initial request must end with a done frame"

    assert dup == [], (
        f"Duplicate (query,k,mode) within 100ms must produce zero frames; got {len(dup)}: {dup}"
    )

    third_hits, third_done = _split_frames(third)
    assert len(third_hits) == 4, (
        f"After waiting past the debounce window, identical request must stream 4 hit frames again, got {len(third_hits)}"
    )
    assert third_done is not None and third_done.get("done") is True, (
        "After waiting past the debounce window, request must end with a done frame"
    )
