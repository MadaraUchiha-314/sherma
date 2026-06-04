# 81 — Managed agent service (hosting sherma agents)

GitHub issue: TBD

## Problem

`sherma` today is a **library** for building declarative agents (YAML or
programmatic, on top of LangGraph + A2A). To run an agent you import the
library, instantiate a `DeclarativeAgent`, and call `send_message()` inside
your own process. There is no way for a user to:

- Onboard to a hosted service without writing/operating Python.
- Author an agent, **publish** an immutable version of it, and **deploy** it.
- Get a stable HTTP endpoint and send **live production traffic** to it.
- Get multi-tenant isolation, durable state, secrets management, observability,
  and billing/metering "for free".

We want a **managed agent service** built on top of the `sherma` library — and
critically, we want the service itself to be **self-hostable** so that others
can stand up their own deployment of managed agents on this stack (cf. how
LangGraph Platform, Letta, and Vertex Agent Engine separate an OSS agent
*library* from a *runtime/platform*).

## Goal

Decide repo strategy and design the end-to-end architecture of a managed agent
service ("sherma Cloud" / `sherma-server`) with:

- A **control plane** (projects/tenants, agent definitions, versioning,
  deployments, secrets, API keys).
- A **data plane** (stateless runtime workers that execute sherma agents,
  durable state, streaming, tool/code execution sandbox).
- Multi-tenancy + isolation, secrets injection, observability, billing.
- A self-hostable distribution (Docker Compose / Helm) and an optional managed
  cloud on top of the same core.

## Scope (this task)

1. Repo-strategy recommendation (separate repo vs same repo).
2. Target architecture: object model, components, data flow, tech stack.
3. Gap analysis: what the `sherma` library needs before it can be hosted.
4. Tenancy/isolation, state/memory, tool sandboxing, secrets, observability,
   billing models — each with a concrete recommendation grounded in how
   Anthropic, OpenAI, LangGraph, Azure AI Foundry, Vertex, Letta, and Bedrock
   AgentCore do it.
5. A phased rollout plan from "library" → "self-hostable platform" → "managed
   cloud".

## Non-goals (this task)

- Writing the service code. This task produces the **plan**; implementation is
  broken into follow-up tasks (see the plan's rollout phases).
- Choosing a final commercial/pricing model (sketched, not finalized).
- A visual builder UI (noted as a later phase, not designed here).

## Chat Iterations

- Initial request (2026-06-04): user asked how to offer sherma as a hosted
  agent service where users onboard, create their own agent, deploy it, and
  send live traffic; whether to use a separate repo or the same repo; that
  others should be able to use the stack to run their own managed-agent
  deployment; and for a detailed architecture informed by how Anthropic and
  OpenAI (and other players) do managed agents. Researched the landscape and
  produced `tasks/plans/81-managed-agent-service.md`.
