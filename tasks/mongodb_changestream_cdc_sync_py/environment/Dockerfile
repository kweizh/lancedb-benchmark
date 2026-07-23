FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        gnupg procps && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    curl -fsSL https://pgp.mongodb.com/server-8.0.asc | \
        gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor && \
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" \
        > /etc/apt/sources.list.d/mongodb-org-8.0.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        mongodb-org mongodb-mongosh && \
    pip3 install --break-system-packages pytest-json-ctrf lancedb pymongo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/project && chmod 0755 /home/user/project

# Idempotent helper that starts a single-node MongoDB replica set on 127.0.0.1:27017.
RUN mkdir -p /var/lib/mongodb /var/log/mongodb && \
    cat >/usr/local/bin/start-mongo.sh <<'EOF' && \
    chmod +x /usr/local/bin/start-mongo.sh
#!/usr/bin/env bash
# Start (if needed) a single-node MongoDB replica set "rs0" and wait for a primary.
DBPATH=/var/lib/mongodb
LOGPATH=/var/log/mongodb/mongod.log
mkdir -p "$DBPATH" /var/log/mongodb

if ! pgrep -x mongod >/dev/null 2>&1; then
    mongod --replSet rs0 --bind_ip 127.0.0.1 --port 27017 \
        --dbpath "$DBPATH" --logpath "$LOGPATH" --fork >/dev/null 2>&1
fi

# Wait for mongod to accept connections.
for _ in $(seq 1 60); do
    if mongosh --quiet --host 127.0.0.1 --eval 'db.runCommand({ping:1}).ok' >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Initiate the replica set only if it has not been initiated yet.
STATUS=$(mongosh --quiet --host 127.0.0.1 --eval 'try { rs.status().ok } catch (e) { 0 }' 2>/dev/null)
if [ "$STATUS" != "1" ]; then
    mongosh --quiet --host 127.0.0.1 --eval \
        'rs.initiate({_id:"rs0",members:[{_id:0,host:"127.0.0.1:27017"}]})' >/dev/null 2>&1
fi

# Wait for a writable primary to be elected.
for _ in $(seq 1 60); do
    ISPRIMARY=$(mongosh --quiet --host 127.0.0.1 --eval 'db.hello().isWritablePrimary' 2>/dev/null)
    if [ "$ISPRIMARY" = "true" ]; then
        exit 0
    fi
    sleep 1
done

echo "ERROR: MongoDB primary was not elected in time" >&2
exit 1
EOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + local MongoDB single-node replica set for a change-stream CDC synchronizer task (Python)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - extra apt packages: gnupg, procps (pgrep) for MongoDB repo setup and the start helper
#   - always required pip3 packages: pytest-json-ctrf
#   - MongoDB Server 8.0 (mongodb-org) + mongosh, run locally as single-node replica set rs0
#   - pip3 packages: lancedb (target library), pymongo (change-stream client)
#   - /usr/local/bin/start-mongo.sh: idempotent helper to start mongod + initiate replica set rs0
#   - /home/user/project: working directory where the agent builds sync.py

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
