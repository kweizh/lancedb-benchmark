FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        "openai>=1.50,<2" \
        "tiktoken>=0.7" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

RUN mkdir -p /workspace /workspace/db /workspace/output && \
    chmod -R 0755 /workspace

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 + OpenAI embedding registry environment for the openai_registry_auto_embed_py task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 with pyarrow==17.0.0 and numpy==1.26.4 (verified working pins)
#   - openai>=1.50,<2 and tiktoken>=0.7 for the OpenAI embedding registry provider
#   - /workspace: project root used by the task
#   - /workspace/db: empty LanceDB database directory (LANCEDB_URI default)
#   - /workspace/output: directory for run.log and registry_results.json
