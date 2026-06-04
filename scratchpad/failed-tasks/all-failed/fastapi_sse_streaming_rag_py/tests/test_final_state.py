import json
import os
import socket

import pytest
import requests
from xprocess import ProcessStarter

PROJECT_DIR = "/home/user/myproject"
EXPECTED_FIXTURE = os.path.join(PROJECT_DIR, ".expected.json")
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"


def _load_fixture():
    with open(EXPECTED_FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def start_app(xprocess):
    class Starter(ProcessStarter):
        name = "fastapi_sse_app"
        args = [
            "uvicorn",
            "app:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ]
        env = os.environ.copy()
        popen_kwargs = {
            "cwd": PROJECT_DIR,
            "text": True,
        }
        timeout = 120
        terminate_on_interrupt = True

        def startup_check(self):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((HOST, PORT)) == 0

    xprocess.ensure(Starter.name, Starter)
    yield
    info = xprocess.getinfo(Starter.name)
    info.terminate()


def _parse_sse(stream):
    """Yield (event, data) tuples by parsing an SSE byte stream from requests.iter_lines."""
    event = None
    data_lines = []
    for raw in stream.iter_lines(decode_unicode=True):
        # iter_lines yields strings (or None for keep-alives); blank line terminates a frame.
        if raw is None:
            continue
        line = raw
        if line == "":
            if event is not None or data_lines:
                yield event, "\n".join(data_lines)
            event = None
            data_lines = []
            continue
        if line.startswith(":"):
            # SSE comment, ignore.
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            payload = line[len("data:"):]
            # Per SSE spec, a single leading space after the colon is part of
            # the framing and should be stripped exactly once.
            if payload.startswith(" "):
                payload = payload[1:]
            data_lines.append(payload)
        # Other fields (id:, retry:, etc.) are ignored.
    if event is not None or data_lines:
        yield event, "\n".join(data_lines)


def test_chat_endpoint_streams_sse_with_expected_sources(start_app):
    fixture = _load_fixture()
    question = fixture["question"]
    expected_sources = fixture["expected_sources"]

    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question},
        stream=True,
        timeout=120,
    )
    try:
        assert resp.status_code == 200, (
            f"POST /chat returned status {resp.status_code}: {resp.text[:500]}"
        )
        ctype = resp.headers.get("Content-Type", "")
        assert ctype.startswith("text/event-stream"), (
            f"Expected Content-Type starting with 'text/event-stream', got: {ctype}"
        )

        frames = list(_parse_sse(resp))
    finally:
        resp.close()

    token_frames = [data for ev, data in frames if ev == "token"]
    done_frames = [data for ev, data in frames if ev == "done"]

    assert len(token_frames) >= 5, (
        f"Expected at least 5 'event: token' frames, got {len(token_frames)}. "
        f"All frames: {frames!r}"
    )

    concatenated = "".join(token_frames)
    assert len(concatenated) >= 20, (
        f"Concatenated token text is suspiciously short ({len(concatenated)} chars): {concatenated!r}"
    )

    assert len(done_frames) == 1, (
        f"Expected exactly one 'event: done' frame, got {len(done_frames)}. "
        f"All frames: {frames!r}"
    )

    try:
        done_payload = json.loads(done_frames[0])
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"'event: done' payload is not valid JSON: {done_frames[0]!r} ({e})"
        )

    assert isinstance(done_payload, dict) and "sources" in done_payload, (
        f"'event: done' JSON must contain 'sources'. Got: {done_payload!r}"
    )
    sources = done_payload["sources"]
    assert isinstance(sources, list), f"'sources' must be a list, got {type(sources)}."
    assert len(sources) == 3, (
        f"'sources' must contain exactly 3 IDs, got {len(sources)}: {sources!r}"
    )
    assert sources == expected_sources, (
        f"Retrieved sources do not match the build-time expected top-3.\n"
        f"  expected: {expected_sources!r}\n"
        f"  got:      {sources!r}"
    )
