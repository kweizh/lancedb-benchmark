FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        openai==1.54.5 \
        httpx==0.27.2 \
        tiktoken && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Bake the seed script into the image.
RUN cat >/home/user/myproject/_seed.py <<'PYEOF'
"""Seed `/home/user/myproject/lancedb` with a 50-row `docs` table covering
five programming-language internals topics (10 documents per topic).

Each document is embedded once at image build time with real OpenAI
`text-embedding-3-small` using the build-time OPENAI_API_KEY ARG.
"""

import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

DB_DIR = "/home/user/myproject/lancedb"
TABLE_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


# 5 topics x 10 documents each = 50 rows.
TOPIC_DOCS = {
    "python_gil": [
        "Python's Global Interpreter Lock (GIL) is a mutex protecting access to Python objects in the CPython interpreter, preventing multiple native threads from executing Python bytecode at once.",
        "Because of the GIL, CPU-bound Python multithreading rarely scales across cores; only one thread holds the interpreter lock and executes bytecode at any given moment.",
        "The GIL exists primarily to simplify CPython's memory management around reference counting; without it, every refcount mutation would need its own fine-grained lock.",
        "I/O-bound Python threads can still benefit from threading because the GIL is released around blocking system calls such as reads, writes, and network operations.",
        "PEP 703 proposes a no-GIL CPython build, replacing reference counts with biased and deferred schemes so multiple threads can run Python bytecode in parallel.",
        "Multiprocessing sidesteps the GIL by spawning separate Python processes, each with its own interpreter and GIL, at the cost of slower inter-process communication.",
        "C extensions that hold the GIL block all other Python threads; well-behaved extensions release the GIL around long CPU loops with Py_BEGIN_ALLOW_THREADS.",
        "Profiling GIL contention with tools like py-spy or yappi often reveals that supposedly threaded Python workloads are serialized through the interpreter lock.",
        "The GIL is a CPython implementation detail; alternative Python runtimes such as Jython and IronPython have no GIL because they use the host VM's threading model.",
        "Some libraries (NumPy, PyTorch) release the GIL while running compiled kernels, allowing genuine parallelism inside vectorized array operations even on stock CPython.",
    ],
    "rust_borrow_checker": [
        "Rust's borrow checker enforces ownership and lifetime rules at compile time, preventing dangling references, data races, and use-after-free bugs without a garbage collector.",
        "The borrow checker permits either one mutable borrow or any number of immutable borrows of a value at a time, statically ruling out aliasing-mutability hazards.",
        "Lifetimes annotate how long references are valid; the borrow checker verifies that no reference outlives the value it points to.",
        "Non-lexical lifetimes (NLL) extended the borrow checker to consider how borrows are actually used, accepting many programs the older lexical analysis rejected.",
        "Common borrow-checker errors like cannot borrow as mutable because it is also borrowed as immutable often indicate aliased state that needs restructuring or interior mutability.",
        "Interior mutability types such as RefCell and Mutex shift borrow checks to runtime, useful when the compile-time borrow checker is too conservative for a valid design.",
        "Rust's move semantics work with the borrow checker: ownership transfers leave the source value invalid, so a moved-from variable cannot accidentally be reused.",
        "Self-referential structs are difficult in safe Rust because the borrow checker cannot generally prove that internal references remain valid across moves.",
        "The Polonius project reimagines the borrow checker as a datalog-style analysis, aiming to accept more sound programs by reasoning about origins of borrows.",
        "Unsafe Rust lets programmers bypass the borrow checker locally, but the surrounding safe API is expected to uphold the same aliasing invariants the checker would enforce.",
    ],
    "go_gc": [
        "Go's garbage collector is a concurrent, tri-color mark-and-sweep collector that runs alongside application goroutines to minimize stop-the-world pause times.",
        "Pause times for the Go GC are typically under a millisecond on modern hardware; the tradeoff is higher CPU overhead from the concurrent marker.",
        "The Go runtime uses a write barrier during garbage collection to track pointer updates so the concurrent marker does not miss reachable objects.",
        "GOGC tunes the trigger ratio between live heap size and new allocations before the next GC cycle; lowering it reduces memory at the cost of more frequent collections.",
        "Go GC pacing is based on a feedback loop that tries to finish the marking phase before the heap grows past the GOGC target.",
        "Allocations in Go are usually thread-local in per-P (processor) caches, which avoids global lock contention during high allocation throughput.",
        "Escape analysis in the Go compiler decides whether a value can live on the goroutine stack or must be heap-allocated and managed by the GC.",
        "Large allocations bypass per-P caches and go straight to the central allocator, which can cause GC pacing surprises in latency-sensitive services.",
        "The Go GC does not perform compaction; long-running services can suffer from memory fragmentation depending on allocation patterns.",
        "Manual hints like runtime.GC() or debug.SetGCPercent let operators trigger or retune the garbage collector for known workload phases.",
    ],
    "javascript_event_loop": [
        "JavaScript's event loop is a single-threaded scheduler that alternates between executing synchronous call-stack work and draining queued asynchronous tasks.",
        "The microtask queue (Promise callbacks, queueMicrotask) is drained after every task and before rendering, giving Promise continuations stronger ordering guarantees than setTimeout.",
        "setTimeout(fn, 0) does not run immediately; it posts a macrotask that runs after the current task and all pending microtasks complete.",
        "Node.js extends the basic event-loop model with explicit phases (timers, poll, check, close) implemented on top of libuv.",
        "Long synchronous work blocks the event loop and freezes the UI in browsers or stalls request handling in Node.js servers.",
        "process.nextTick in Node.js runs before any I/O or timer callback, even before regular Promise microtasks; misuse can starve the event loop.",
        "Async/await syntax is built on Promises and the microtask queue; an awaited Promise resumes the suspended function as a microtask once the Promise settles.",
        "The browser's requestAnimationFrame callback is scheduled to run before the next paint, not on the microtask or macrotask queues used by Promises or timers.",
        "Web Workers move CPU-heavy work off the main event loop into a separate thread with its own event loop, communicating via postMessage.",
        "Starvation of the event loop manifests as dropped frames in the browser or rising response latencies in Node.js services under load.",
    ],
    "java_jit": [
        "The Java HotSpot JIT compiler converts bytecode to native code at runtime, using profiling data to specialize hot methods for the observed workload.",
        "HotSpot uses tiered compilation: methods are first interpreted, then compiled by C1 for fast warmup, and finally by C2 for peak optimization.",
        "Inlining is one of the JIT's most impactful optimizations; small hot methods are inlined into their callers to enable further optimization across the boundary.",
        "Escape analysis in the JIT can eliminate heap allocations for short-lived objects that do not escape their method, replacing them with stack or scalar storage.",
        "Deoptimization happens when a JIT-compiled method's speculative assumption is invalidated; the runtime falls back to the interpreter and may recompile later.",
        "GraalVM's JIT compiler is a polyglot replacement for C2 that uses an alternative IR and often produces faster code on specific workloads.",
        "Ahead-of-time compilation with jaotc or native-image trades runtime adaptivity for predictable startup and removes the JIT warmup phase entirely.",
        "Code cache exhaustion in the JIT can disable further compilation, causing previously hot methods to revert to the interpreter and lose performance.",
        "Diagnostic flags like -XX:+PrintCompilation and -XX:+PrintInlining expose the JIT's decisions to help tune Java performance on production workloads.",
        "On-stack replacement (OSR) lets the JIT compile a long-running loop while it is executing, jumping into native code without waiting for the next method call.",
    ],
}


def build_rows():
    rows = []
    next_id = 0
    for topic in sorted(TOPIC_DOCS):
        docs = TOPIC_DOCS[topic]
        assert len(docs) == 10, (topic, len(docs))
        for d in docs:
            rows.append({"id": next_id, "topic": topic, "content": d})
            next_id += 1
    assert len(rows) == 50, len(rows)
    return rows


def embed_batch(client, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for item in resp.data:
            out.append(item.embedding)
    return out


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("[seed] ERROR: OPENAI_API_KEY is required at build time.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DB_DIR, exist_ok=True)

    rows = build_rows()
    texts = [r["content"] for r in rows]

    client = OpenAI()
    print(f"[seed] Embedding {len(texts)} documents with {EMBED_MODEL}...", flush=True)
    embeddings = embed_batch(client, texts)
    assert len(embeddings) == len(rows), (len(embeddings), len(rows))
    for r, vec in zip(rows, embeddings):
        assert len(vec) == EMBED_DIM, (r["id"], len(vec))
        r["embedding"] = vec

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("topic", pa.string()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
    ])
    arrays = {
        "id": [int(r["id"]) for r in rows],
        "topic": [r["topic"] for r in rows],
        "content": [r["content"] for r in rows],
        "embedding": [np.asarray(r["embedding"], dtype=np.float32) for r in rows],
    }
    table_arrow = pa.table(arrays, schema=schema)

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=table_arrow, mode="create")
    n_rows = tbl.count_rows()
    print(f"[seed] Table '{TABLE_NAME}' now has {n_rows} rows.", flush=True)
    assert n_rows == 50, n_rows
    # Avoid lancedb 0.25.3 SIGABRT at interpreter teardown after heavy writes.
    os._exit(0)


main()
PYEOF

# Build-time seed: requires --build-arg OPENAI_API_KEY=...
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
RUN test -n "$OPENAI_API_KEY" && \
    python3 /home/user/myproject/_seed.py
# Scrub the key out of the image layer; runtime will inject a fresh value.
ENV OPENAI_API_KEY=

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB query-decomposition RAG task: 50-row docs table (5 programming-language internals topics x 10 docs) seeded at build time with real OpenAI text-embedding-3-small (1536-d). Candidate builds decompose / decomposed_search / baseline_search and must demonstrate that decomposed retrieval covers strictly more topics than the single-shot baseline."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 + pyarrow==17.0.0 + numpy==1.26.4 + pandas==2.2.3 + openai==1.54.5 + httpx==0.27.2 + tiktoken
#   - /home/user/myproject/_seed.py: build-time seed script (already executed during image build)
#   - /home/user/myproject/lancedb/docs: 50-row LanceDB table (5 topics x 10 docs) with 1536-d real OpenAI embeddings
