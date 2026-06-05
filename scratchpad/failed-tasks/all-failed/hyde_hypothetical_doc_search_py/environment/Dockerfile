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
RUN mkdir -p /app

# Bake the seed script into the image. It builds a 30-row corpus of detailed
# paragraphs about programming languages. One row (the "rigged" doc) is a deep
# cross-language essay about garbage collection that the deliberately short
# query "GC differences between languages?" will NOT retrieve in vector-only
# search because the query is too lexically sparse and the corpus also contains
# a surface-level "differences between languages" distractor. HyDE rescues it:
# an LLM-written hypothetical paragraph about GC mechanics across languages
# embeds very close to the rigged doc.
#
# The seed script verifies BOTH invariants at build time:
#   * baseline_search_top1 != rigged_id  (the short query must miss it)
#   * hyde_search_top1     == rigged_id  (gpt-4o-mini hypothetical recovers it)
# If either invariant fails the build fails fast.
RUN cat >/app/_seed.py <<'PYEOF'
"""Seed /app/lancedb_data with a 30-row programming_qa table.

Embeds all 30 detailed paragraphs with real OpenAI text-embedding-3-small (1536-d).
Also calls real OpenAI gpt-4o-mini at temperature=0 to generate a hypothetical
answer to the deliberately ambiguous query, verifies the HyDE invariant, and
writes /app/.expected.json for the verifier.
"""

import json
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

DB_DIR = "/app/lancedb_data"
EXPECTED_JSON = "/app/.expected.json"
TABLE_NAME = "programming_qa"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBED_DIM = 1536

QUERY = "GC differences between languages?"

# The rigged document is a long, detailed cross-language essay on garbage
# collection mechanics. It deliberately uses canonical GC vocabulary
# ("reference counting", "generational", "tricolor", "mark-and-sweep",
# "stop-the-world", "ownership") that a competent LLM hypothetical will
# also produce, so the HyDE-expanded query lands directly on this row.
RIGGED_DOC = {
    "id": 7,
    "language": "multi",
    "text": (
        "Memory reclamation strategies vary substantially across modern programming "
        "languages. CPython relies primarily on per-object reference counting "
        "augmented by a generational cycle detector for unreachable strongly "
        "connected components. The Java HotSpot virtual machine ships several "
        "tracing collectors (Parallel, G1, ZGC, Shenandoah) based on generational "
        "hypotheses, concurrent marking, and remembered sets so that most pauses "
        "are short. Go uses a concurrent tricolor mark-and-sweep collector with "
        "write barriers that runs alongside goroutines and aims for sub-millisecond "
        "stop-the-world phases. V8's JavaScript engine combines a generational "
        "scavenger for the young generation with an incremental mark-and-sweep "
        "major collector, and it also performs idle-time compaction. Rust takes a "
        "fundamentally different route: it has no runtime tracing collector at all, "
        "because its compile-time ownership and borrowing rules statically prove "
        "when each value can be dropped, freeing memory deterministically at scope "
        "exit. Erlang and Elixir, running on the BEAM virtual machine, give each "
        "lightweight process its own private heap and collect those heaps "
        "independently, eliminating global pauses and shared mutable state. "
        "Choosing among these designs is a tradeoff between throughput, pause "
        "latency, memory overhead, and the cognitive cost imposed on the "
        "programmer at compile time."
    ),
}

# A surface-level distractor that matches the query phrase "differences between
# languages" but does NOT discuss garbage collection. Pure vector search on the
# very short query token-set will gravitate toward this row instead of the
# rigged GC essay above.
DISTRACTOR_DOC = {
    "id": 12,
    "language": "multi",
    "text": (
        "Programming languages differ along many dimensions beyond the obvious "
        "syntactic ones. Differences between languages show up in their type "
        "systems (static versus dynamic, structural versus nominal), in their "
        "execution model (compiled versus interpreted versus JIT), in their "
        "paradigm support (object-oriented, functional, procedural, logic), in "
        "their concurrency primitives (threads, coroutines, channels, actors), "
        "in their tooling ecosystems, in their package managers and registries, "
        "and in cultural conventions around naming, formatting, and testing. When "
        "evaluating differences between languages for a new project teams usually "
        "weigh hiring availability, performance characteristics, library coverage, "
        "interop with existing systems, and the size and maturity of the community."
    ),
}

OTHER_DOCS = [
    {"id": 0, "language": "python", "text": (
        "Python is a high-level dynamically typed language with significant "
        "whitespace, duck typing, and a famously readable syntax. It supports "
        "multiple paradigms including procedural, object-oriented and functional "
        "styles. CPython's global interpreter lock serializes bytecode execution "
        "across threads which makes CPU-bound multithreading ineffective, so "
        "library authors typically reach for multiprocessing or asyncio instead. "
        "The standard library is unusually rich and the third-party ecosystem on "
        "PyPI covers data science, web frameworks, and machine learning."
    )},
    {"id": 1, "language": "java", "text": (
        "Java is a statically typed object-oriented language that compiles to JVM "
        "bytecode and runs on the Java Virtual Machine. Its type system uses "
        "nominal classes and interfaces, generics with type erasure, and explicit "
        "checked exceptions. The JVM provides just-in-time compilation, profiling, "
        "and adaptive optimization, which is why long-running Java services often "
        "approach the throughput of natively compiled code. The Java ecosystem is "
        "centered on Maven and Gradle for builds and on a vast catalog of "
        "enterprise libraries for HTTP, persistence, and messaging."
    )},
    {"id": 2, "language": "c", "text": (
        "C is a small, portable, statically typed systems language. It exposes raw "
        "pointers, manual memory management with malloc and free, and undefined "
        "behavior as a first-class concept that compilers exploit aggressively for "
        "optimization. C has no built-in module system, no exceptions, no string "
        "type beyond null-terminated character arrays, and no concurrency "
        "primitives in the language itself. Decades of operating systems, "
        "embedded firmware, language runtimes, and databases are written in C, and "
        "its calling convention is the de facto FFI lingua franca."
    )},
    {"id": 3, "language": "cpp", "text": (
        "C++ extends C with classes, templates, RAII, move semantics, lambdas, and "
        "a rich standard library of containers and algorithms. Resource management "
        "is deterministic via destructors and smart pointers rather than tracing "
        "collection, which gives predictable performance but demands careful "
        "ownership thinking. Templates and constexpr enable extensive "
        "compile-time computation. The language is famously feature-rich and the "
        "modern subset that revolves around values, move semantics, and standard "
        "ranges produces concise, fast code."
    )},
    {"id": 4, "language": "rust", "text": (
        "Rust is a statically typed systems language whose distinguishing feature "
        "is its ownership and borrowing model. Every value has a single owner, and "
        "the borrow checker enforces at compile time that references never outlive "
        "the data they point to, which eliminates use-after-free and data race "
        "bugs without a runtime tracing collector. Rust has algebraic data types, "
        "pattern matching, traits with associated types, and zero-cost "
        "abstractions. Cargo, its build tool and package manager, is widely "
        "considered one of the most pleasant in the industry."
    )},
    {"id": 5, "language": "go", "text": (
        "Go is a statically typed compiled language designed at Google for "
        "large-scale services. Its syntax is intentionally small, its standard "
        "library covers networking and concurrency comprehensively, and its "
        "concurrency model is built around goroutines and channels following CSP. "
        "Go favors composition over inheritance via small interfaces that are "
        "satisfied structurally. The toolchain produces single self-contained "
        "binaries and offers gofmt for canonical formatting, which has effectively "
        "ended bike-shedding about style in Go codebases."
    )},
    {"id": 6, "language": "javascript", "text": (
        "JavaScript is the language of the web browser and, via Node.js, of much "
        "of the modern server tier. It is dynamically typed, prototype-based, and "
        "single-threaded around an event loop that drives non-blocking I/O. "
        "Modern JavaScript (ES2015 and later) added classes, modules, arrow "
        "functions, destructuring, generators, and async/await on top of "
        "Promises. Browsers and Node share a common language but differ in "
        "their host APIs."
    )},
    # id 7 reserved for RIGGED_DOC
    {"id": 8, "language": "typescript", "text": (
        "TypeScript is a syntactic superset of JavaScript that adds an optional, "
        "gradual, structural type system. It is erased at build time: the tsc "
        "compiler emits plain JavaScript that runs in any JS host. TypeScript's "
        "type system supports generics, conditional types, mapped types, and "
        "discriminated unions, and it is expressive enough to encode many domain "
        "invariants that would otherwise live in runtime tests. Adoption in the "
        "frontend ecosystem is now nearly ubiquitous."
    )},
    {"id": 9, "language": "ruby", "text": (
        "Ruby is a dynamically typed object-oriented language with a strong "
        "emphasis on programmer happiness and expressive metaprogramming. Blocks, "
        "method_missing, and open classes let library authors build internal "
        "DSLs that read like English. The standard MRI implementation has a "
        "global VM lock, much like CPython's GIL, which makes Ruby threads "
        "cooperative for I/O but not parallel for CPU work. Rails, the dominant "
        "Ruby web framework, popularized convention-over-configuration design."
    )},
    {"id": 10, "language": "php", "text": (
        "PHP is a server-side scripting language designed originally for HTML "
        "templating and now used by a large slice of the web. Modern PHP is "
        "dynamically typed but supports optional scalar and return type hints, "
        "attributes, enums, and short closures. Composer is the dominant package "
        "manager and frameworks like Laravel and Symfony provide modern MVC "
        "stacks. Long-running PHP processes are uncommon: the typical deployment "
        "is a per-request fresh interpreter behind FPM or a web server module."
    )},
    {"id": 11, "language": "swift", "text": (
        "Swift is a statically typed, modern language developed by Apple and used "
        "primarily for iOS, macOS, watchOS, and server applications. It features "
        "value types via structs, protocol-oriented programming, optionals to "
        "encode nullability in the type system, and automatic reference counting "
        "for memory management. Swift's syntax is concise and its standard "
        "library is built around protocols and generics."
    )},
    # id 12 reserved for DISTRACTOR_DOC
    {"id": 13, "language": "kotlin", "text": (
        "Kotlin is a statically typed JVM language created by JetBrains that "
        "interoperates seamlessly with Java. It eliminates several Java pain "
        "points by making nullability part of the type system, providing data "
        "classes, sealed classes, extension functions, and structured concurrency "
        "via coroutines. Kotlin is the preferred language for Android development "
        "and is increasingly used on the server with Spring Boot and Ktor."
    )},
    {"id": 14, "language": "scala", "text": (
        "Scala fuses object-oriented and functional programming on the JVM. It "
        "has a rich static type system with implicits, higher-kinded types, "
        "pattern matching, and case classes. The ecosystem includes Akka for "
        "actor-based concurrency, Spark for distributed data processing, and "
        "Play for web services. Scala 3 simplified the syntax and replaced "
        "implicits with given and using clauses to make the type machinery "
        "easier to teach."
    )},
    {"id": 15, "language": "haskell", "text": (
        "Haskell is a purely functional, statically typed, lazily evaluated "
        "language. Side effects are explicitly tracked in the type system via "
        "monads such as IO, State, and STM, which lets the compiler reason "
        "about purity. The type system supports parametric polymorphism, type "
        "classes, GADTs, and rank-N types. Lazy evaluation gives Haskell its "
        "famous ability to compose infinite data structures cleanly but also "
        "introduces memory-leak footguns around thunks."
    )},
    {"id": 16, "language": "elixir", "text": (
        "Elixir is a dynamically typed functional language built on the BEAM "
        "virtual machine, the same runtime as Erlang. It uses lightweight "
        "preemptively scheduled processes that communicate by message passing, "
        "and supervisor trees give a principled approach to fault tolerance. "
        "Elixir's macro system and pipe operator make data transformation "
        "pipelines very readable, and the Phoenix framework is the dominant "
        "choice for web applications."
    )},
    {"id": 17, "language": "ocaml", "text": (
        "OCaml is a statically typed functional language with a powerful module "
        "system, type inference via Hindley-Milner, algebraic data types, and "
        "pattern matching. It compiles to fast native code and is widely used "
        "in formal methods, financial trading systems, and language tooling. "
        "OCaml's module system, with functors that take and return modules, is "
        "considered one of the most expressive in mainstream languages."
    )},
    {"id": 18, "language": "clojure", "text": (
        "Clojure is a Lisp dialect that runs on the JVM and embraces immutable "
        "persistent data structures as a default. Concurrency is handled by a "
        "small set of orthogonal primitives: atoms for uncoordinated synchronous "
        "updates, refs with software transactional memory for coordinated "
        "updates, and agents for asynchronous updates. ClojureScript targets "
        "the browser, and the language is known for its emphasis on simple "
        "values over complex objects."
    )},
    {"id": 19, "language": "erlang", "text": (
        "Erlang is a dynamically typed functional language designed at Ericsson "
        "for telecom switches and famous for nine-nines availability in real "
        "systems. It runs on the BEAM virtual machine, uses preemptively "
        "scheduled lightweight processes that communicate by message passing, "
        "and supports hot code reloading. Its 'let it crash' philosophy combined "
        "with supervisor trees produces remarkably resilient distributed "
        "applications."
    )},
    {"id": 20, "language": "lua", "text": (
        "Lua is a small, fast, dynamically typed scripting language designed to "
        "be embedded inside larger applications. Its primary data structure is "
        "the table, a versatile associative array used to model arrays, "
        "records, namespaces, and objects. Lua is widely used as the scripting "
        "layer in games, embedded systems, and network appliances. LuaJIT is a "
        "tracing just-in-time compiler that delivers performance competitive "
        "with much heavier runtimes."
    )},
    {"id": 21, "language": "perl", "text": (
        "Perl is a dynamically typed scripting language whose original niche "
        "was text processing and system administration. It has first-class "
        "regular expressions, list and hash context, and an unusually flexible "
        "syntax. CPAN, the Comprehensive Perl Archive Network, is one of the "
        "oldest and largest package ecosystems in existence. Perl is still "
        "actively maintained and continues to be used heavily in "
        "bioinformatics and legacy infrastructure."
    )},
    {"id": 22, "language": "r", "text": (
        "R is a dynamically typed language for statistical computing and "
        "graphics. Its core data structure is the data frame, and its standard "
        "library is rich in distributions, hypothesis tests, and modeling "
        "functions. The tidyverse, a coherent set of packages including ggplot2, "
        "dplyr, and tidyr, defines the modern idiomatic R style. R is widely "
        "used in academic statistics, epidemiology, econometrics, and "
        "exploratory data analysis."
    )},
    {"id": 23, "language": "julia", "text": (
        "Julia is a dynamically typed but type-stable language designed for "
        "high-performance numerical and scientific computing. It is built around "
        "multiple dispatch: methods are selected by the runtime types of all "
        "arguments. A just-in-time compiler based on LLVM produces native code "
        "and, given type-stable functions, Julia can match C and Fortran on "
        "benchmark numerical workloads. The package ecosystem covers "
        "differential equations, optimization, and probabilistic programming."
    )},
    {"id": 24, "language": "dart", "text": (
        "Dart is a statically typed language developed by Google and used "
        "primarily through the Flutter framework to build cross-platform "
        "mobile, web, and desktop applications. It compiles ahead-of-time to "
        "native code for production and uses a just-in-time compiler during "
        "development for fast hot reload. Sound null safety is baked into the "
        "type system, and the language has classes, mixins, generics, and "
        "async-await with futures and streams."
    )},
    {"id": 25, "language": "zig", "text": (
        "Zig is a small, statically typed systems language that aspires to be a "
        "spiritual successor to C. It has no hidden control flow and no hidden "
        "memory allocations: allocators are always passed explicitly. comptime "
        "lets the same syntax express compile-time computation and generics, "
        "and there is a built-in cross-compiler. The build system is itself a "
        "Zig program, which gives unusually flexible build configuration."
    )},
    {"id": 26, "language": "fortran", "text": (
        "Fortran is the oldest high-level language still in active use, dating "
        "to 1957 and continuously revised. It is statically typed, with deep "
        "compiler support for array operations and parallel constructs such as "
        "DO CONCURRENT and coarrays. Fortran dominates legacy and modern "
        "high-performance computing codes in climate modeling, computational "
        "chemistry, and finite-element analysis where decades of validated "
        "numerical kernels are difficult to replicate."
    )},
    {"id": 27, "language": "cobol", "text": (
        "COBOL is a statically typed business-oriented language designed in 1959. "
        "Its syntax reads almost like English and was intended to be readable by "
        "non-programmers. COBOL is built around fixed-format records, decimal "
        "arithmetic, and batch file processing, and it still runs enormous "
        "amounts of banking, insurance, and government workloads on IBM "
        "mainframes. Modern variants add object-oriented features and run on "
        "the JVM and .NET."
    )},
    {"id": 28, "language": "assembly", "text": (
        "Assembly languages map essentially one-to-one to the instruction set "
        "of a particular processor architecture. Programming directly in "
        "assembly is rare outside operating system kernels, embedded firmware, "
        "performance hotspots in cryptography or video codecs, reverse "
        "engineering, and compiler backends. Modern assemblers offer macros, "
        "conditional assembly, and structured directives, but the programmer "
        "is still responsible for registers, the calling convention, and "
        "instruction-level scheduling."
    )},
    {"id": 29, "language": "lisp", "text": (
        "Common Lisp is a dynamically typed multi-paradigm dialect of Lisp "
        "standardized by ANSI in the 1990s. It is famous for its macro system, "
        "which lets programmers extend the language with new control structures "
        "indistinguishable from the built-ins, and for its interactive "
        "development model in which a running image is incrementally edited at "
        "the REPL. CLOS, the Common Lisp Object System, supports multiple "
        "dispatch and method combination."
    )},
]


def _build_corpus():
    rows = OTHER_DOCS + [RIGGED_DOC, DISTRACTOR_DOC]
    rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), f"Duplicate ids: {ids}"
    assert ids == list(range(30)), f"Ids must be 0..29 dense, got {ids}"
    return rows


def _embed_batch(client, texts, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for item in resp.data:
            out.append(item.embedding)
    return out


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("[seed] ERROR: OPENAI_API_KEY is required at build time.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DB_DIR, exist_ok=True)

    rows = _build_corpus()
    rigged_id = RIGGED_DOC["id"]
    distractor_id = DISTRACTOR_DOC["id"]

    client = OpenAI()
    texts = [r["text"] for r in rows]
    print(f"[seed] Embedding {len(texts)} docs with {EMBED_MODEL}...", flush=True)
    doc_embeddings = _embed_batch(client, texts)
    assert len(doc_embeddings) == len(rows), (len(doc_embeddings), len(rows))
    for r, vec in zip(rows, doc_embeddings):
        assert len(vec) == EMBED_DIM, (r["id"], len(vec))
        r["vector"] = vec

    # -------------------- Build-time HyDE invariant check --------------------
    # Baseline invariant: short query top-1 != rigged_id.
    qresp = client.embeddings.create(model=EMBED_MODEL, input=[QUERY])
    qvec = qresp.data[0].embedding
    sims = sorted(
        ((r["id"], _cosine(qvec, r["vector"])) for r in rows),
        key=lambda kv: kv[1],
        reverse=True,
    )
    baseline_top = [rid for rid, _ in sims[:5]]
    print(f"[seed] baseline (raw query) top-5 ids: {baseline_top}", flush=True)
    assert baseline_top[0] != rigged_id, (
        f"Fixture invariant FAILED: baseline top-1 must NOT be the rigged doc, "
        f"but got {baseline_top[0]} == rigged_id={rigged_id}. The whole point of "
        "HyDE is to recover a doc the baseline misses."
    )

    # HyDE invariant: gpt-4o-mini (temperature=0) hypothetical -> top-1 == rigged_id.
    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "Please write a detailed paragraph answering this question:"},
            {"role": "user", "content": QUERY},
        ],
    )
    hypothetical = chat.choices[0].message.content
    assert isinstance(hypothetical, str) and len(hypothetical) >= 100, (
        f"gpt-4o-mini returned an implausibly short hypothetical: {hypothetical!r}"
    )
    print(f"[seed] hypothetical chars={len(hypothetical)}", flush=True)
    hresp = client.embeddings.create(model=EMBED_MODEL, input=[hypothetical])
    hvec = hresp.data[0].embedding
    hsims = sorted(
        ((r["id"], _cosine(hvec, r["vector"])) for r in rows),
        key=lambda kv: kv[1],
        reverse=True,
    )
    hyde_top = [rid for rid, _ in hsims[:5]]
    print(f"[seed] HyDE top-5 ids: {hyde_top}", flush=True)
    assert hyde_top[0] == rigged_id, (
        f"Fixture invariant FAILED: HyDE top-1 must equal rigged_id={rigged_id}, "
        f"got {hyde_top[0]}. Top-5 was {hyde_top}. The rigged GC essay must "
        "out-rank every other doc against an LLM-generated answer."
    )

    # ------------------------ Write the LanceDB table ------------------------
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ])
    arrays = {
        "id": [int(r["id"]) for r in rows],
        "language": [r["language"] for r in rows],
        "text": [r["text"] for r in rows],
        "vector": [np.asarray(r["vector"], dtype=np.float32) for r in rows],
    }
    table_arrow = pa.table(arrays, schema=schema)

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=table_arrow, mode="create")
    n_rows = tbl.count_rows()
    print(f"[seed] Table '{TABLE_NAME}' now has {n_rows} rows.", flush=True)
    assert n_rows == 30, n_rows

    # ------------------------- Write the fixture file ------------------------
    with open(EXPECTED_JSON, "w") as f:
        json.dump(
            {
                "rigged_id": int(rigged_id),
                "distractor_id": int(distractor_id),
                "query": QUERY,
                "baseline_top5": [int(x) for x in baseline_top],
                "hyde_top5": [int(x) for x in hyde_top],
            },
            f,
            indent=2,
        )
    print(f"[seed] Wrote {EXPECTED_JSON}.", flush=True)
    # Avoid lancedb 0.25.3 SIGABRT at interpreter teardown after writes.
    os._exit(0)


main()
PYEOF

# Build-time seed: requires --build-arg OPENAI_API_KEY=...
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
RUN test -n "$OPENAI_API_KEY" && \
    python3 /app/_seed.py
# Scrub the key out of the image layer; runtime will inject a fresh value.
ENV OPENAI_API_KEY=

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + HyDE retrieval task: 30-row programming-language corpus pre-embedded at build time with real OpenAI text-embedding-3-small (1536-d cosine). The rigged GC essay row is invisible to the short query 'GC differences between languages?' under vector-only search but is recovered as rank-1 when an LLM-written hypothetical paragraph (gpt-4o-mini, temperature=0) is embedded instead."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 + pyarrow==17.0.0 + numpy==1.26.4 + pandas==2.2.3 + openai==1.54.5 + httpx==0.27.2 + tiktoken
#   - /app/_seed.py: build-time seed script (already executed during image build)
#   - /app/lancedb_data/programming_qa: 30-row LanceDB table with 1536-d real OpenAI embeddings
#   - /app/.expected.json: rigged_id + query + baseline/HyDE top-5 fixture
#   - /home/user/myproject: empty candidate project directory
