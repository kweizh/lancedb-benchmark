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
        tiktoken \
        markdown-it-py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject

# Bake the documentation corpus into the image at /app/docs/.
#
# Six Markdown files, each with exactly one '# Title' and several '## Section'
# headers. The verifier asks three queries, each rigged so that exactly one
# section is unambiguously the top-1 answer:
#
#   - auth-guide.md   / "SSO Setup"               -> Okta SAML SSO configuration
#   - api-reference.md / "Rate Limiting"          -> HTTP 429 + Retry-After + backoff
#   - migrations.md   / "Rolling Back Migrations" -> rolling back a failed DB migration
#
# Words such as "Okta", "SAML", "429", "Retry-After", and the exact phrase
# "rolling back a failed database migration" appear in exactly one section
# across the entire corpus, which is what makes the anchor sections deterministic
# winners for OpenAI semantic search.
RUN mkdir -p /app/docs && \
    cat >/app/docs/auth-guide.md <<'EOF'
# Auth Guide

## Overview

This guide describes how authentication is wired up in our platform. We support
three distinct authentication modes: local username and password accounts, OAuth
based social login through Google and GitHub, and enterprise single sign-on
through a SAML-compatible identity provider. The choice of mode is per-tenant
and may be changed at any time by an administrator. Throughout the guide we
refer to the user-facing service as the application and to the authentication
service as the identity broker.

## SSO Setup

To configure single sign-on with Okta SAML, navigate to the Okta admin console,
create a new SAML 2.0 application, and copy the metadata XML URL. Paste that
URL into the application admin console under Settings -> Identity Provider
-> SAML. The Okta SAML integration uses the standard SAML 2.0 web browser SSO
profile with HTTP-POST binding. You must configure the audience restriction to
match your application entity ID and map the Okta user attributes nameid, email
and groups onto the corresponding claims used by the application. Once Okta
returns a signed SAML assertion, the identity broker validates the signature
against the Okta certificate, provisions a session, and redirects the user to
the originally requested URL. Okta SAML SSO is the recommended option for any
customer with an existing Okta tenant.

## Local Accounts

Local accounts are stored directly in the application database and are intended
for small deployments or pilots where setting up an identity provider is not
worthwhile. Each local account stores an email, a bcrypt-hashed password, and
an opaque per-user salt. Password complexity rules are enforced at sign-up time
and at every password change. We do not recommend local accounts for production
deployments at companies that already have a corporate identity provider.

## Password Recovery

If a user forgets their password, they can request a recovery email from the
sign-in page. The application generates a single-use recovery token, stores its
SHA-256 hash in the database with a fifteen minute expiry, and sends the token
to the user as part of a recovery link. When the user follows the link, the
application verifies the hash, prompts for a new password, updates the bcrypt
hash, and invalidates any outstanding recovery tokens for that account.
EOF
RUN cat >/app/docs/api-reference.md <<'EOF'
# API Reference

## REST Endpoints

The public REST API is served at https://api.example.com/v1 and is documented
using OpenAPI 3.1. The most commonly used resources are /users, /projects,
/jobs, and /reports. Every response is JSON encoded with UTF-8. Request bodies
must be JSON encoded as well, and the Content-Type header must be set to
application/json. Pagination follows the cursor pattern: responses include a
next_cursor field, which the client passes back as the cursor query parameter
on subsequent requests.

## Webhooks

Webhooks let an external service receive notifications when a resource changes.
A webhook is identified by its target URL and a shared secret. The application
delivers a JSON payload with an HMAC-SHA256 signature in the X-Signature
header. If the receiving server returns a non-2xx status, the application
queues the delivery for a later retry using a fixed schedule, up to a maximum
of twelve delivery attempts over twenty four hours. Webhook subscriptions are
managed through the /webhooks resource of the REST API.

## Rate Limiting

The API enforces request quotas per API key. When a client exceeds its quota,
the server responds with HTTP status 429 Too Many Requests and includes a
Retry-After header indicating the number of seconds the client must wait
before issuing the next request. Clients are required to honor the
Retry-After value. We strongly recommend that clients implement exponential
backoff with jitter when they observe repeated 429 responses, doubling the
wait time after each consecutive throttled request. The exact quotas are
documented per endpoint in the OpenAPI specification and can also be inspected
at runtime through the X-RateLimit-Limit and X-RateLimit-Remaining response
headers.

## Authentication Headers

Every API request must carry an Authorization header in the form
"Authorization: Bearer <token>". The token is issued by the identity broker
described in the auth guide. Tokens are short lived and must be refreshed
using the standard OAuth 2.0 refresh token flow. Requests without a valid
Authorization header receive an HTTP 401 Unauthorized response.
EOF
RUN cat >/app/docs/deployment.md <<'EOF'
# Deployment Manual

## Kubernetes Deployment

The application ships as a set of Kubernetes manifests under deploy/k8s. Each
microservice is packaged as a Deployment object with a paired Service and
HorizontalPodAutoscaler. We assume the target cluster runs Kubernetes 1.28 or
later and has an Ingress controller already installed. Container images are
pulled from our private registry and rolled out using the default Kubernetes
RollingUpdate strategy.

## Bare Metal Setup

For customers running on bare metal hardware, we provide an Ansible playbook
that configures the operating system, installs the required dependencies, and
brings up the application using systemd unit files. The bare metal deployment
path is intended for restricted environments where Kubernetes is not available
and for smaller, single-host installations.

## Rolling Updates

Rolling updates are performed by gradually replacing old replicas with new
replicas while the workload remains available. The Kubernetes RollingUpdate
strategy is parameterized by maxSurge and maxUnavailable. For typical
deployments we recommend setting both to 25 percent, which trades a small
amount of additional resource usage during the rollout for a smaller window
of degraded capacity. The kubectl rollout status command can be used to
monitor progress.

## Rollback Strategy

If a new release of the application introduces a regression at the service
level, you can revert to the previous container image using kubectl rollout
undo on the affected Deployment. Kubernetes keeps the previous revision history
by default and rolling back a service Deployment to the previous revision is
typically completed in under a minute. This section is concerned only with
reverting container images for running services; data store schema rollbacks
are described in the database migrations guide and are not in scope here.
EOF
RUN cat >/app/docs/performance.md <<'EOF'
# Performance Tuning

## CPU Profiling

To diagnose CPU bottlenecks, attach the py-spy profiler to a running worker and
collect a flame graph over a representative load period of at least sixty
seconds. Look for functions that contribute more than five percent of total
CPU time and inspect their call sites. Common offenders include synchronous
JSON encoding inside tight loops, regular expression compilation inside hot
paths, and accidental quadratic algorithms over input lists.

## Memory Optimization

Resident set size is tracked per pod and graphed under the Memory dashboard.
If a service exceeds its memory budget, the first step is to capture a heap
snapshot using tracemalloc. The most common causes of high memory usage are
unbounded caches, large pandas DataFrames held in module scope, and accidental
references to closed file objects. Replace unbounded caches with size-limited
or time-limited alternatives.

## Database Indexing

Slow queries are almost always the result of a missing index. Use the database
EXPLAIN ANALYZE command to confirm that a query is using a sequential scan
and then add the appropriate index. Composite indexes should be ordered with
the most selective column first. Be aware that adding an index increases the
cost of inserts and updates, so add indexes only when the corresponding
queries are observed in production load.

## Caching Strategies

For read-heavy workloads, place a Redis cache in front of the database and
adopt a cache-aside pattern: read from the cache, fall back to the database
on a miss, and write the database value back into the cache with a TTL.
Choose a TTL that balances staleness against load reduction. Avoid caching
values that are personalized per user unless the cache key includes the user
identifier.
EOF
RUN cat >/app/docs/monitoring.md <<'EOF'
# Monitoring and Alerts

## Metrics Collection

All services expose Prometheus metrics on the /metrics endpoint. The metrics
follow the four golden signals: latency, traffic, errors, and saturation.
Service owners are expected to add custom business metrics on top of the
default runtime metrics provided by the application framework. Prometheus
scrapes every fifteen seconds and Grafana dashboards visualize the metrics.

## Log Aggregation

Application logs are written to stdout in structured JSON format and shipped
to a central log aggregation system using Fluent Bit. Each log line carries a
trace identifier so that logs can be correlated with distributed traces. Retain
logs for thirty days online and ninety days in cold storage.

## PagerDuty Integration

Critical alerts are routed to PagerDuty using the events v2 API. Each alert
includes a routing key tied to the on-call schedule for the responsible
service. Non-critical notifications are routed to Slack rather than PagerDuty
to avoid alert fatigue. Use the PagerDuty incident workflow to coordinate
incident response and document the resolution timeline.

## Dashboard Configuration

Dashboards are stored in source control as JSON files under monitoring/grafana
and are automatically loaded by Grafana on startup. Every service has a
golden-signals dashboard and a dependency-overview dashboard. Use the
dashboard naming convention service-name/golden-signals so that consumers can
predict the URL of a dashboard for any given service.
EOF
RUN cat >/app/docs/migrations.md <<'EOF'
# Database Migrations

## Schema Versioning

The application uses Alembic to manage database schema changes. Every change
is captured as a Python migration script under db/migrations and tagged with a
revision identifier. The current schema version is stored in the alembic_version
table. Migration scripts must be checked in alongside the corresponding code
changes so that each commit is internally consistent.

## Forward Migrations

A forward migration moves the schema from the previous revision to the next
revision. Forward migrations should be additive whenever possible: add new
columns and tables, leave existing columns in place, and only drop columns
after the deploying release has been confirmed safe in production. Forward
migrations are applied automatically as part of the standard deploy pipeline
once the new release reaches the canary stage.

## Rolling Back Migrations

The procedure to roll back a failed database migration that left the schema in
an inconsistent state is as follows. First, pause the deploy pipeline so that
no further forward migrations run while you investigate. Second, identify the
last known-good revision id from the alembic_version table. Third, run
"alembic downgrade <revision_id>" against the affected database, which executes
the downgrade callback of every revision newer than the target. If a downgrade
callback errors, fix the data inconsistency manually using a one-off SQL
script, mark the affected revision as downgraded by updating the
alembic_version table, and rerun the downgrade command. Once the database is
back at the known-good revision, the application can be redeployed and the
forward migration retried with the data fix in place. This procedure is the
only supported way to recover from a failed schema migration; do not attempt
to edit the alembic_version table without a downgrade unless you fully
understand the implications.

## Data Backfills

When a forward migration introduces a new column whose value depends on
existing rows, schedule a data backfill job to populate the column after the
schema change has been applied. Backfills are written as idempotent batch jobs
that process at most ten thousand rows at a time and resume from a checkpoint
so they can be safely retried.
EOF

# Sanity check: confirm the docs are present and each has exactly one '# title'
# and at least 3 '## section' headers.
RUN python3 - <<'PYEOF'
import os

DOCS = "/app/docs"
expected = ["auth-guide.md", "api-reference.md", "deployment.md", "performance.md", "monitoring.md", "migrations.md"]
names = sorted(os.listdir(DOCS))
assert set(names) == set(expected), (names, expected)
for name in expected:
    with open(os.path.join(DOCS, name), encoding="utf-8") as f:
        text = f.read()
    title_lines = [ln for ln in text.splitlines() if ln.startswith("# ") and not ln.startswith("## ")]
    section_lines = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert len(title_lines) == 1, (name, title_lines)
    assert len(section_lines) >= 3, (name, section_lines)
# Anchor phrases must appear in exactly one section of exactly one doc.
import re
def count(phrase):
    n = 0
    for name in expected:
        with open(os.path.join(DOCS, name), encoding="utf-8") as f:
            text = f.read()
        if phrase.lower() in text.lower():
            n += 1
    return n
assert count("Okta SAML") >= 1, "Okta SAML must appear in auth-guide.md"
assert count("429") == 1, "HTTP 429 must appear in exactly one file"
assert count("Retry-After") == 1, "Retry-After must appear in exactly one file"
assert count("roll back a failed database migration") == 1, (
    "anchor phrase 'roll back a failed database migration' must appear in exactly one file"
)
assert count("alembic downgrade") == 1, (
    "alembic downgrade anchor must appear only in migrations.md"
)
print("docs corpus sanity OK")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB Markdown-docs-repo indexer task. /app/docs/ contains 6 baked markdown files with one '# title' and several '## sections' each; verifier issues 3 rigged semantic queries."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#           openai==1.54.5 httpx==0.27.2 tiktoken markdown-it-py
#   - /app/docs/{auth-guide,api-reference,deployment,performance,monitoring,migrations}.md
#   - /home/user/myproject                  (empty project dir for the candidate)
