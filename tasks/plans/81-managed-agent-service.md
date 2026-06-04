# Plan: Managed agent service (hosting sherma agents)

**Task:** [tasks/81-managed-agent-service.md](../81-managed-agent-service.md)
**GitHub Issue:** TBD

---

## 0. TL;DR (the recommendation in one paragraph)

Keep `sherma` as a clean, permissively-licensed **agent engine library** and
build the managed service as a **separate repository** (`sherma-server`) that
depends on `sherma`. Mirror the proven split used by every serious player:
an **OSS agent library** + a **separately-shipped runtime/platform** (LangGraph
OSS + LangGraph Platform; Google ADK + Vertex Agent Engine; OpenAI Agents SDK +
AgentKit; Letta lib + Letta server). Design the platform around a **control
plane / data plane** architecture with an **Agent → Version → Deployment →
Thread → Run → Event** object model (a blend of Anthropic Managed Agents'
Agent/Environment/Session/Events and OpenAI's immutable-version-on-publish).
The data plane is **stateless runtime workers + Redis (queue + pub/sub) +
Postgres (durable state)** — the LangGraph/n8n pattern — with tool/code
execution pushed into **per-session sandboxes**. Ship the whole platform as a
**self-hostable** Docker Compose / Helm bundle first (so others can run their
own managed-agent deployment), and offer a managed cloud on top of the exact
same core.

---

## 1. Background & Analysis

### 1.1 What sherma is today (the engine we're hosting)

From the current codebase:

- `DeclarativeAgent(id, version, yaml_path|yaml_content, tenant_id, hooks,
  checkpointer, http_async_client)` loads a YAML manifest and compiles a
  LangGraph graph. `LangGraphAgent` is the programmatic base class.
- Execution is **async + streaming**: `async for event in
  agent.send_message(request)` yields A2A `Message` / `TaskStatusUpdateEvent`
  (incl. `input_required` for human-in-the-loop interrupts) / `Task` /
  artifact events.
- **Multi-tenancy primitives already exist**: `tenant_id` on every entity, a
  `TenantRegistryManager`, per-tenant `RegistryBundle` (Prompt/LLM/Tool/Skill/
  Agent registries) — task 8 already built tenant-scoped registries.
- **State/threads**: LangGraph checkpointer keyed by `thread_id` (derived from
  A2A `context_id`/`task_id`). Default is `MemorySaver` (in-process only).
- **Extensibility**: 20 lifecycle hooks; hooks can run **remotely** over
  JSON-RPC / MCP (tasks 12, 77) — important, it means custom logic doesn't have
  to run in the core worker process.
- **Tools/skills**: LangChain tools, MCP servers, progressive-disclosure skills,
  sub-agents-as-tools, A2A remote agents.
- **Observability**: Langfuse integration already present.

**Implication:** sherma is already a strong *data-plane runtime*. The biggest
missing pieces are everything *around* it — a control plane, durable state,
deployment/orchestration, secrets, auth, sandboxing, metering — plus a few
runtime hardening items (Postgres checkpointer, graph caching, building an agent
from *stored* config rather than a file path, per-tenant secret injection).

### 1.2 How the major players do "managed agents" (research synthesis)

The full cited research is summarized here; sources in §13.

**Anthropic** runs a two-tier model:
- *Agent SDK* (library, runs in your infra; one session = one subprocess;
  pluggable `SessionStore` mirroring transcripts to S3/Redis/Postgres; they
  explicitly recommend per-session sandbox containers — Modal/E2B/Daytona/Fly/
  Firecracker/gVisor).
- *Managed Agents* (hosted REST+SSE) with a clean object model:
  **Agent** (config: model+prompt+tools+MCP+skills, referenced by ID) /
  **Environment** (managed *or* self-hosted sandbox) / **Session** (running
  instance) / **Events** (the message stream). State is server-side; tools run
  either server-side (MCP connector) or via client callback. Credential
  isolation pattern: authenticate at a gateway in front of the agent; inject
  outbound tool credentials at an egress proxy, never in the agent env.

**OpenAI**: Responses API (primitive) + Agents SDK (OSS, runs in your process:
Agents/Handoffs/Guardrails/Sessions) + AgentKit (hosted Agent Builder/ChatKit).
Key transferable idea: **publishing a workflow creates an immutable, pinnable
version snapshot**. State via `previous_response_id` chaining or durable
Conversation objects. Tracing is **on by default**. (Note the moving targets:
Assistants API sunsets 2026-08-26; Agent Builder is being deprecated — so don't
copy the *visual builder* as a durable surface, copy the *versioning* model.)

**LangGraph Platform** — the closest blueprint:
- Explicit **control plane** (stores desired state: deployments, revisions,
  assistants) vs **data plane** (the running LangGraph Servers + a *listener*
  that polls the control plane and reconciles, Kubernetes-style).
- **Assistants** = named, versioned, parameterized configs of a graph (exactly
  the indirection sherma needs between a YAML manifest and a deployed instance).
- Runtime = **Postgres (durable state/checkpoints) + Redis (queue/broker +
  per-run pub/sub for streaming)** with **stateless queue workers** (atomic
  `BLPOP`, at-least-once).
- Three deployment tiers: Cloud SaaS / Hybrid (SaaS control plane + data plane
  in your VPC) / fully self-hosted.
- **Licensing caveat**: `langgraph` lib is MIT but the `langgraph-api` server
  runtime is **Elastic License 2.0** (needs a commercial key in prod). We will
  *not* depend on `langgraph-api`; we build our own thin server around the OSS
  LangGraph library that sherma already uses.

**Others worth copying specific ideas from:**
- *Azure AI Foundry Agent Service* — best **tenancy** model: **basic** =
  provider-managed shared storage with logical separation; **standard** =
  bring-your-own storage (Cosmos/Blob/Search/Key Vault) in the customer's tenant
  + VNet/private endpoints for hard isolation.
- *Vertex Agent Engine / ADK* — clean **two-tier memory**: Sessions (events +
  short-term state) vs Memory Bank (managed, per-scope-isolated long-term
  memory). REST `:query` endpoints, auto-managed sessions.
- *CrewAI AMP* — declarative `agents.yaml`/`tasks.yaml`; async **`/kickoff` +
  poll `/status` + webhook event streaming** API shape.
- *Letta* (Apache-2.0) — state-first: everything persisted in Postgres, agent
  survives restarts; runs as an agent server with a REST API. Most reusable
  license if we want to borrow code.
- *Bedrock AgentCore* — gold-standard **isolation**: one session = one
  **Firecracker microVM**, destroyed + memory-sanitized at session end;
  framework-agnostic (bring your own container).
- *Durable execution substrates* (Temporal / Restate / Inngest) — the
  build-vs-buy fork for crash-safe, resumable, exactly-once runs. LangGraph
  rolled its own (Postgres checkpoints + Redis); we can start the same way and
  keep the door open to adopt Temporal/Restate later.

### 1.3 The recurring reference architecture

Every platform converges on the same component set:

1. Declarative, **versioned** agent definitions + an artifact/config registry,
   with an indirection layer between *template* and *deployed instance*.
2. **Control plane** (desired state, APIs/UI) vs **data plane** (running
   workers); often a reconciling listener between them.
3. **Stateless workers + Redis (queue/broker/pub-sub) + Postgres (durable
   state)**; optionally a durable-execution engine underneath.
4. **Durable/resumable** runs (checkpoint at step boundaries; HITL pause/resume).
5. **Two-tier state**: short-term thread/session state vs long-term cross-session
   memory. Postgres is the de-facto session store.
6. **Multi-tenancy in two grades**: shared-with-logical-separation vs
   BYO-storage/VPC; per-session microVM for untrusted tool/code execution.
7. **Observability as a first-class plane** (traces of LLM/tool/agent calls,
   tokens/cost/latency) + webhook/streaming for live updates.
8. **Deployment spectrum**: Cloud SaaS → Hybrid → fully self-hosted.
9. **API shape** converging on async run/kickoff + poll status + stream +
   webhooks, REST per deployment.

---

## 2. Repo strategy — decision

**Decided (2026-06-04): a single `uv` **workspace monorepo** with two
separately-published packages**, and an **Apache-2.0** license so others can
self-host their own managed-agent deployment. Proposed layout:

```
sherma/                      # repo root (uv workspace)
  packages/
    sherma/                  # the agent ENGINE library (this package, moved under packages/)
    sherma-server/           # the PLATFORM: control + run API, workers, adapters, deploy artifacts
  pyproject.toml             # [tool.uv.workspace] members = ["packages/*"]
```

| Package | What it is | License |
|---|---|---|
| `sherma` | The agent **engine** library. Stays focused, importable, easy to test, **no web/DB/infra deps**. Published to PyPI as today. | Apache-2.0 |
| `sherma-server` | The **platform**: control-plane API, data-plane workers, persistence adapters, sandbox runner, dashboard, deploy artifacts (Compose/Helm). Depends on `sherma` (workspace dep). | **Apache-2.0** — no "no-competing-service" clause, so the stack is genuinely reusable (the explicit goal). |

**Why the workspace monorepo (vs two repos or one package):**

1. **Clean dependency direction is preserved** — `sherma-server → sherma`, never
   the reverse. The library must stay free of Postgres/Redis/FastAPI/K8s
   concerns. Some *runtime extension points* the platform needs (Postgres
   checkpointer, graph caching, build-agent-from-config, `SecretProvider`)
   belong **in the library** as optional, dependency-light hooks; see Phase 1.
2. **Separate release/versioning is retained** — each package versions and
   publishes independently; library consumers still `pip install sherma` without
   pulling any platform deps.
3. **No cross-repo coordination tax** — a server change that needs a new library
   hook is one PR across both packages (uv resolves the workspace dep locally),
   avoiding the two-repo "PR + release + bump" round-trip during fast iteration.
4. **Self-hostability stays clean** — `sherma-server` keeps its own
   Compose/Helm, docs, and version; people `git clone` the repo and run the
   platform, or `pip install sherma` for just the library.

> Migration note: moving the existing `sherma` package under `packages/sherma`
> is a mechanical but repo-wide move (imports stay the same; paths in CI,
> `pyproject.toml`, and tooling configs change). Do it as the first step of
> Phase 0 in its own PR to keep the diff reviewable.

---

## 3. Target architecture

### 3.1 Object model (control-plane entities)

Blend of Anthropic (Agent/Environment/Session/Events) + OpenAI (immutable
version on publish) + LangGraph (assistant/thread/run), mapped onto sherma:

| Entity | Description | Maps to sherma |
|---|---|---|
| **Project** (a.k.a. Workspace / Tenant) | Billing + isolation boundary; owns API keys, members, secrets, quotas. | `tenant_id` |
| **Agent** | A named logical agent within a project. | — (new) |
| **AgentVersion** | **Immutable snapshot** of a YAML manifest (+ pinned tool/skill/MCP refs), created on *publish*. Validated against the sherma schema at publish time. | The YAML `manifest` + `DeclarativeConfig` |
| **Environment** | Where runs execute + which secrets/credentials/MCP connectors/sandbox profile they get. Managed sandbox vs BYO. | env vars / `http_async_client` / secret injection |
| **Deployment** | A running, addressable binding of `AgentVersion × Environment` with scaling config. Exposes a stable endpoint + A2A AgentCard. Has *revisions*. | A live `DeclarativeAgent` instance pool |
| **Thread** (Session) | A conversation; durable short-term state. | checkpointer `thread_id` / A2A `context_id` |
| **Run** | One invocation against a thread (sync / async / streaming). | one `send_message()` stream |
| **Event** | Stream items for a run (messages, status, artifacts, tool calls). | A2A events |
| **Memory** (later) | Long-term, cross-thread memory store, per-scope isolated. | new (Vertex Memory Bank style) |

Publishing is snapshot-on-publish (OpenAI model): `AgentVersion` is immutable;
`Deployment` pins a version and can be re-pointed/rolled back instantly.

### 3.2 Components & topology

```
                         ┌────────────────────────── CONTROL PLANE ──────────────────────────┐
   Users / Dashboard ───▶│  Control API (FastAPI)                                             │
   CI / SDK / CLI        │   • Projects, Agents, AgentVersions (publish/validate)             │
                         │   • Deployments & revisions (desired state)                        │
                         │   • Secrets (write-only → KMS/Vault), API keys, members, quotas    │
                         │   • Postgres (control DB) + audit log                              │
                         └───────────────▲───────────────────────────────┬───────────────────┘
                                         │ desired state / reconcile      │ traces, usage
                                         │                                ▼
   ┌──────── EDGE ────────┐    ┌──────────────────────── DATA PLANE ─────────────────────────┐
   │ API Gateway          │    │  Run API / Ingress (auth, rate-limit, route by deployment)   │
   │ • authN (API key/JWT)│───▶│        │                                                     │
   │ • per-tenant rate    │    │        ▼ enqueue                                             │
   │ • request routing    │    │   Redis  ──(BLPOP)──▶  Runtime Workers (stateless)           │
   └──────────────────────┘    │   • run queue          • build/cache sherma agent from       │
                               │   • per-run pub/sub       AgentVersion (+ secret injection)  │
   live traffic ──────────────▶│   • streaming chan       • agent.send_message() loop         │
                               │                          • publish events → Redis pub/sub    │
                               │                                   │ tool/code calls          │
                               │                                   ▼                          │
                               │                          Sandbox layer (per-session)         │
                               │                          • Firecracker/microVM | gVisor |    │
                               │                            E2B/Modal/Daytona | egress proxy  │
                               │                          • MCP tunnels to private servers     │
                               │                                                              │
                               │  Postgres (data DB): threads, checkpoints, runs, events,     │
                               │  artifacts metadata    •  Object store (S3): files/artifacts │
                               └──────────────────────────────────────────────────────────────┘
                                         │ OTEL spans + token/cost metering
                                         ▼
                         Observability + Billing (Langfuse/OTEL collector, usage meter)
```

### 3.3 Request/data flow (a live run)

1. Client calls `POST /v1/deployments/{id}/runs` (or `/threads/{tid}/runs`) with
   an input message; gateway authenticates the project API key, checks quota,
   routes to the Run API.
2. Run API resolves the deployment → `AgentVersion` + `Environment`, creates a
   `Run` row (Postgres), and **enqueues** the job on Redis.
3. A stateless **runtime worker** pops the job, **builds/looks-up a cached
   sherma agent** for that `AgentVersion` (graph caching — see Phase 1),
   attaches the **Postgres checkpointer** for the thread, and **injects
   secrets** (tool/LLM creds) for that environment into the agent's resolution
   context (never persisted).
4. Worker runs `async for event in agent.send_message(...)`, **publishing each
   event to a Redis pub/sub channel** keyed by run id and **persisting** events
   + final checkpoint to Postgres. Tool/code execution is dispatched into the
   **per-session sandbox**; outbound calls go through the **egress proxy** that
   injects credentials and enforces per-tenant allowlists.
5. The client's `GET /runs/{id}/stream` (SSE) subscribes to the Redis channel
   for token/event streaming; async clients poll `GET /runs/{id}`; optional
   **webhooks** fire on status changes (CrewAI-style).
6. Interrupts (`input_required`) surface as a run status; the next run on the
   same thread resumes from the checkpoint. Tokens/cost are metered to the
   project; OTEL spans go to the observability plane.

This is the LangGraph/n8n runtime pattern (stateless workers + Redis +
Postgres), with Anthropic's gateway-auth + egress-credential-injection security
model and AgentCore-style per-session sandbox isolation layered on.

---

## 4. Gap analysis — what the `sherma` library needs first

These are dependency-light additions that belong **in the library** (Phase 1),
because the platform builds on them and other library users benefit too:

1. **Durable checkpointer.** Replace/augment `MemorySaver` with a
   `PostgresSaver`-backed option (LangGraph already provides `langgraph-checkpoint-postgres`).
   Expose via the existing `checkpointer` injection point — minimal change.
2. **Build agent from stored config, not just a file path.** Today
   `DeclarativeAgent` takes `yaml_path` or `yaml_content`. The platform stores
   manifests in a DB and references skills/prompts by URL. Confirm/extend
   `yaml_content` + remote refs path so an agent can be constructed entirely
   from in-memory config + registry URLs (no local filesystem). Mostly already
   supported (remote hooks/MCP/skills exist); needs an explicit "no local FS"
   path + base-path-free resolution.
3. **Graph caching / agent pooling.** `get_graph()` runs per `send_message()`.
   For hosting, cache the compiled graph per `(tenant, agent_id, version)` so
   workers don't recompile on every request. Add a cache hook or make compiled
   graphs reusable across invocations (thread isolation is already via
   checkpointer `thread_id`, so a compiled graph is safe to share).
4. **Secret provider abstraction.** Today LLM/tool creds come from process env
   vars. Add a pluggable `SecretProvider` (or resolver) so the platform can
   inject per-tenant/per-environment secrets at agent-build time without
   touching global env. Keep env-var provider as the default.
5. **Resource/limits hooks.** Surface timeouts, max steps/tokens, and
   cancellation cleanly (some exist: `max_concurrency`, `cancel_task`). Ensure
   `cancel_task` reliably tears down in-flight tool/sandbox work.
6. **(Optional) durable-execution seam.** Keep the run loop structured so it can
   later sit on Temporal/Restate without rewrites (idempotent step boundaries,
   externalized side-effects). Not required for MVP.

Everything else (control plane, queue, sandbox, gateway, billing, dashboard)
lives in `sherma-server`.

---

## 5. Multi-tenancy & isolation

Adopt **two grades** (Azure AI Foundry model), and **per-session sandboxing**
for code/tool execution (AgentCore model):

- **Logical isolation (default / cloud multi-tenant).** Reuse sherma's existing
  `tenant_id` + `TenantRegistryManager`. Every control-plane row and data-plane
  artifact is tenant-scoped; Postgres row-level scoping by `project_id`; Redis
  keys namespaced by project; rate limits + quotas per project.
- **Hard isolation (enterprise / BYO).** "Standard setup": customer brings their
  own Postgres / object store / secret store / VPC; data-plane workers run in the
  customer's network (Hybrid deployment tier). Control plane stays managed.
- **Execution isolation.** Untrusted tool/code execution and Bash-like tools
  must not run in the shared worker process. Run them in a **per-session
  sandbox** — start pragmatic (gVisor/containers or a managed provider like
  E2B/Modal/Daytona), with a path to **Firecracker microVM per session**
  (AgentCore-grade) for the strongest tenants. MCP servers: prefer
  HTTP/streamable transport over stdio for multi-tenancy; support **MCP tunnels**
  for reaching private customer MCP servers.
- **Egress credential isolation.** Outbound tool credentials are injected at an
  **egress proxy**, not stored in the agent/worker env; per-tenant egress
  allowlists (Anthropic hosting guidance).

---

## 6. State & memory

Two-tier, Postgres-backed (industry default):

- **Short-term / thread state** = sherma checkpointer with `PostgresSaver`,
  keyed by `thread_id`. Durable, resumable, HITL-pausable (already supported via
  LangGraph interrupts + A2A `input_required`). Threads are project-scoped.
- **Long-term / cross-thread memory** (later phase) = a managed **Memory** store
  with per-scope isolation (Vertex Memory Bank / Letta archival model). sherma's
  skill/data mechanisms + a new memory tool can surface it to agents.
- **Event log** persisted in Postgres (Anthropic Managed Agents store events
  server-side; fetchable history) + object storage for large artifacts/files.
- **Retention controls from day one** — server-side state means data-residency
  and deletion APIs matter (Anthropic flags ZDR/HIPAA implications). Per-project
  retention policy + hard-delete endpoints for threads/runs/files.

---

## 7. Security & secrets

- **Front-door auth at the gateway**: project-scoped API keys (hashed at rest)
  and/or JWT; the agent/worker receives pre-authenticated requests.
- **Secrets**: write-only API → KMS/Vault/Secrets Manager; resolved into the
  agent build via the `SecretProvider` hook (Phase 1); never logged, never
  persisted in checkpoints/events. Per-environment scoping.
- **Egress proxy** for outbound tool/LLM calls with per-tenant allowlists +
  credential injection.
- **Sandbox** for tool/code execution (see §5).
- **Tenant config-leak hardening** for any shared-process execution: no implicit
  filesystem config loading; per-tenant working dirs; (mirrors Anthropic's
  `settingSources: []` / disable-auto-memory multi-tenant guidance).
- **Audit log** of control-plane mutations.

---

## 8. Observability & billing

- **Observability** as a first-class plane. sherma already integrates Langfuse;
  standardize on **OTEL** spans for LLM/tool/agent/node calls (sherma's hook
  system is the natural emission point — a built-in tracing hook executor).
  Provide a per-project trace view (LangSmith/Langfuse-style). Default-on
  tracing for good DX (OpenAI model), with opt-out + sensitive-data exclusion.
- **Streaming/webhooks** for live run updates (Redis pub/sub → SSE; webhooks on
  run status).
- **Billing/metering**: meter token usage (input/output/cache), run counts,
  sandbox-hours, storage. Tokens dominate cost (Anthropic's own note), so the
  metering hook must capture per-run token + tool-second usage and roll up per
  project. Wire to a usage table → invoicing/limits.

---

## 9. Self-hostability ("let others run their own managed-agent deployment")

This is an explicit requirement, so design for it from the start:

- **One `docker compose up`** brings up: control API, run API/gateway, a worker,
  Postgres, Redis, object store (MinIO), and an OTEL/Langfuse collector.
- **Helm chart** for Kubernetes (workers as a scalable Deployment; gateway as a
  Service/Ingress).
- **Pluggable backends** behind interfaces: state store (Postgres), queue/broker
  (Redis), object store (S3-compatible), secret store (env/Vault/KMS), sandbox
  provider (local/gVisor/E2B/Modal/Firecracker), auth (API key/OIDC). Defaults
  that run locally; swap for cloud in production.
- **Permissive license** on `sherma-server` (no "no-competing-service" clause)
  so the stack is genuinely reusable — the opposite of n8n SUL / Dify's clause /
  LangGraph's ELv2 server.
- **Three deployment tiers** documented (LangGraph model): self-hosted / hybrid
  (managed control plane + customer-VPC data plane) / managed cloud.

---

## 10. Tech stack (proposed for `sherma-server`)

- **Language/runtime**: Python 3.13 + `uv` (consistent with sherma); async.
- **Control & Run API**: FastAPI + Pydantic (sherma already uses Pydantic;
  reuse `DeclarativeConfig` for validation on publish).
- **DB / migrations**: Postgres + SQLAlchemy/SQLModel + Alembic; LangGraph
  `PostgresSaver` for checkpoints.
- **Queue / streaming**: Redis (queue via lists/streams + pub/sub for run
  streaming). Keep a seam to swap for Temporal/Restate later.
- **Object store**: S3-compatible (MinIO for local).
- **Sandbox**: start with containers/gVisor or a managed provider (E2B/Modal/
  Daytona); roadmap to Firecracker.
- **Secrets**: env → Vault/cloud KMS via `SecretProvider`.
- **Observability**: OTEL + Langfuse.
- **Transport to agents**: reuse sherma's **A2A** server (`A2AStarletteApplication`
  + `ShermaAgentExecutor`) so every deployment also speaks the A2A protocol and
  can be consumed by other agents — a differentiator (native agent-to-agent).
- **Dashboard** (later): Next.js or reuse/extend the existing Streamlit UI for
  an MVP console.

---

## 11. Phased rollout plan

Each phase is a shippable increment; later phases are separate sherma /
sherma-server tasks.

**Phase 0 — Decisions & scaffolding.** Confirm repo strategy + license (§2,
§12). Create `sherma-server` repo (or uv workspace). Stand up CI mirroring
sherma's (ruff/pyright/pytest). Compose skeleton (Postgres+Redis+MinIO).

**Phase 1 — Library runtime hardening (in `sherma`).** Postgres checkpointer
option; build-agent-from-stored-config (no-FS path); graph caching / agent
reuse; `SecretProvider` abstraction; robust cancellation. *Each of these is a
normal sherma feature task and per CLAUDE.md must update `docs/` +
`skills/sherma/references/` + `SKILL.md`.*

**Phase 2 — Single-tenant platform MVP (self-hostable).** Control API (Agents,
AgentVersions w/ publish+validate, Deployments). Run API: create/stream/poll
runs against a deployment; Redis queue + stateless worker running sherma;
Postgres-persisted threads/runs/events; SSE streaming. `docker compose up`
works end-to-end. A2A endpoint per deployment. (Single tenant = "default".)

**Phase 3 — Multi-tenancy, auth, secrets.** Projects + API keys + gateway auth;
tenant-scoped everything (reuse `TenantRegistryManager`); secret store +
injection; per-project quotas/rate limits; audit log; retention/delete APIs.

**Phase 4 — Sandboxed execution & connectors.** Per-session sandbox for
tool/code execution; egress proxy + per-tenant allowlists; MCP over HTTP +
private MCP tunnels; managed vs BYO environments.

**Phase 5 — Observability, billing, console.** OTEL tracing hook + per-project
trace UI; webhooks; usage metering (tokens/sandbox-hours/storage) + quotas/
limits; MVP dashboard.

**Phase 6 — Long-term memory + enterprise + managed cloud.** Memory store
(per-scope isolation); Hybrid deployment tier (customer-VPC data plane); Helm
chart hardening; optional Firecracker isolation; launch managed cloud on the
same core. Evaluate moving the run loop onto Temporal/Restate if durability
needs outgrow the Redis+Postgres approach.

### Docs & skill updates (CLAUDE.md compliance)

- Library-affecting work (Phase 1, and any later library hooks) updates `docs/`,
  `skills/sherma/references/`, and `SKILL.md` in the **same task**, per the repo
  rule. Plans for those sub-tasks must include the docs+skill step explicitly.
- Platform-only work lives in `sherma-server` with its own docs; it does **not**
  change sherma agent behavior, so it does not require sherma `docs/`/skill
  edits — except where it drives a new library extension point, which does.

---

## 12. Decisions & remaining open questions

**Resolved (2026-06-04):**

1. **Repo layout** → single **uv-workspace monorepo** (`packages/sherma` +
   `packages/sherma-server`). See §2.
2. **Platform license** → **Apache-2.0** (permissive, honors "let others
   self-host"). See §2.
3. **Primary audience** → **both, equally**: build the self-hostable core first
   (Phase 2), then layer cloud + enterprise on the same base. Keep the current
   phase ordering rather than skewing toward dev-SaaS or enterprise-first.

**Still open (decide before the relevant phase):**

4. **Durable execution**: roll our own (Redis+Postgres, LangGraph-style) for MVP
   vs adopt Temporal/Restate up front. Recommendation: roll our own first, keep
   the seam. (Decide by Phase 2.)
5. **Sandbox provider** for v1: managed (E2B/Modal/Daytona — fastest) vs
   self-run gVisor/Firecracker (more control, more ops). (Decide by Phase 4.)

---

## 13. References (research sources)

Anthropic: [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview),
[Hosting the Agent SDK](https://code.claude.com/docs/en/agent-sdk/hosting),
[Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview),
[Skills guide](https://platform.claude.com/docs/en/build-with-claude/skills-guide),
[MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector),
[Agent capabilities API](https://www.anthropic.com/news/agent-capabilities-api),
[MCP Tunnels (InfoQ)](https://www.infoq.com/news/2026/05/claude-mcp-tunnels/).

OpenAI: [New tools for building agents](https://openai.com/index/new-tools-for-building-agents/),
[Agents SDK](https://openai.github.io/openai-agents-python/),
[Tools](https://openai.github.io/openai-agents-python/tools/),
[Guardrails](https://openai.github.io/openai-agents-python/guardrails/),
[Tracing](https://openai.github.io/openai-agents-python/tracing/),
[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state),
[Introducing AgentKit](https://openai.com/index/introducing-agentkit/),
[Agent Builder deprecation](https://community.openai.com/t/deprecation-notice-agent-builder/1382650),
[Deprecations](https://developers.openai.com/api/docs/deprecations).

LangGraph: [Control plane](https://langchain-ai.github.io/langgraph/concepts/langgraph_control_plane/),
[Data plane](https://docs.langchain.com/langgraph-platform/data-plane),
[Persistence](https://docs.langchain.com/oss/python/langgraph/persistence),
[Redis/queue internals](https://neuralware.github.io/posts/langgraph-redis/),
[Platform GA](https://blog.langchain.com/langgraph-platform-ga/),
[ELv2 licensing note](https://rvernica.github.io/2026/03/langchain-license).

Others: CrewAI [deploy to AMP](https://docs.crewai.com/en/enterprise/guides/deploy-to-amp),
[webhook streaming](https://docs.crewai.com/en/enterprise/features/webhook-streaming);
Azure [standard agent setup](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/concepts/standard-agent-setup),
[Cosmos DB integration](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/azure-agent-service);
Vertex [sessions](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview),
[Memory Bank](https://cloud.google.com/agent-builder/agent-engine/memory-bank/overview);
[Letta](https://github.com/letta-ai/letta);
[n8n queue mode](https://docs.n8n.io/hosting/scaling/queue-mode/);
durable execution [Temporal](https://temporal.io/), [Restate](https://docs.restate.dev/foundations/key-concepts),
[Inngest](https://www.inngest.com/blog/durable-execution-key-to-harnessing-ai-agents);
isolation [AgentCore runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html),
[isolated sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html).

## Plan Revisions

- **2026-06-04**: Locked in the previously-open repo/license/audience decisions
  after user input. Repo strategy (§2) changed from "separate repo recommended,
  monorepo as fallback" to **uv-workspace monorepo** (`packages/sherma` +
  `packages/sherma-server`) as the chosen approach, with an added migration note
  about moving the existing package under `packages/`. Platform license fixed at
  **Apache-2.0**. Audience set to **both equally** (self-hostable core first).
  §12 reorganized into resolved vs still-open decisions (durable-execution
  engine and sandbox provider remain open).
