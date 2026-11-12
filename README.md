# PaperLens

PaperLens is a multimodal research workspace for ingesting scientific PDFs, reading
their structured content, running hybrid retrieval, asking citation-grounded
questions, discovering related work, and executing bounded LangGraph research
workflows.

## Repository layout

```text
src/
  backend/
    app/
      main.py, routes.py, cli.py, config.py, schemas.py, auth.py
      db/              SQLAlchemy models and repositories
      harness/         agent runtime, conversation context, guardrails
      tools/           coding and math LangChain tools
      ingestion/       Docling parse, clean, assets, chunk, enrich
      rag/             embeddings, retrieval, grounded QA
      research/        discovery, compare, LangGraph workflow
      infrastructure/  storage, accelerator, observability
    tests/      backend test suite
    scripts/    parsing, evaluation and database utilities
    evaluation/ retrieval and QA datasets/results
    migrations/ database bootstrap SQL
  frontend/     static HTML/CSS/JavaScript research workspace
  ops/          AWS deployment and secret-management utilities
infra/
  bootstrap/    Terraform remote-state bucket (one-time)
  prod/         Terraform for the AWS production stack
runtime/
  data/         local SQLite data
  outputs/      local paper artifacts
.env.example    environment template
Dockerfile.*    CPU API, GPU API and frontend images
```

## Local setup

Requirements: Python 3.11+, Git, and optionally Docker.

```powershell
git clone https://github.com/phamanhquan12/PaperLens.git
cd PaperLens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".\src\backend[dev]"
Copy-Item .env.example .env
```

Fill the provider and database values in `.env`. Never commit `.env`.

### Required AI provider credentials

PaperLens does not include shared LLM credentials. Every person or organization
self-hosting PaperLens must provide credentials from an OpenAI-compatible
provider in their own `.env`:

```text
LLM_ENABLED=true
LLM_API_KEY=your-provider-key
LLM_MODEL=your-chat-model
LLM_BASE_URL=https://your-provider.example/v1

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your-provider-key
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_BASE_URL=https://your-provider.example/v1
```

Visual and cleaned-text enrichment are optional. To enable them, configure
`LUNA_API_KEY`, `LUNA_MODEL`, and `LUNA_BASE_URL` using your own provider
account. A self-hosted production deployment must never reuse credentials from
the public PaperLens deployment or commit provider keys to Git.
Set `INGEST_AUTO_VISUAL_ENRICH=true` to run Luna on every table, figure and
formula during upload (one VLM request per element); its descriptions are
added, labelled as model-generated, to the search index. After changing
chunking or enrichment settings, rebuild existing papers with
`python -m app.cli reindex [--paper-id ID] [--enrich]`.

Retrieval is small-to-big: short sections are merged into parent passages of
800–1800 tokens, only their children (≤450 tokens) and the table/figure/formula
chunks are searched, and the LLM receives the parent passages of the matches.
Table chunks include the table data; figure/table context is taken from the
sentences that reference them ("as shown in Table 2").

These keys belong only in the backend `.env` or a cloud secret manager; never
put them in frontend configuration. Visitors opening an already-hosted
PaperLens URL do not edit `.env` and currently consume that deployment owner's
configured provider account. Per-user bring-your-own-key (BYOK) is not yet
implemented.

### Optional account mode

PaperLens supports Supabase email/password authentication with server-enforced
paper and conversation ownership. Run `src/backend/migrations/002_accounts_and_sessions.sql`
against an existing database before enabling it, then configure:

```text
AUTH_ENABLED=true
SUPABASE_AUTH_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
SUPABASE_JWKS_URL=https://YOUR_PROJECT.supabase.co/auth/v1/.well-known/jwks.json
```

The publishable key is injected only into the frontend container. PaperLens
verifies access tokens against Supabase JWKS and does not require a secret or
service-role key. Existing pre-v0.2 rows remain assigned to `local-user` and
must be reassigned or removed before production account mode is enabled.

When account mode is enabled, visitors can also choose **Continue without an
account**. That creates a browser-bound guest session with hard limits
(default: 10 chat queries, 2 paper uploads, 2 image uploads) that consume the
deployment owner's configured LLM keys. Guest papers and chats stay isolated
from signed-in accounts. Set `GUEST_TRIAL_ENABLED=false` to disable the option.

## Run locally

Backend:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend in another terminal:

```powershell
$env:PAPERLENS_API_URL = "http://127.0.0.1:8000"
python -m http.server 3000 --directory src\frontend
```

Open `http://127.0.0.1:3000`.

## Tests

```powershell
python -m pytest src\backend\tests -m "not integration" -q
```

Optional sample ingestion:

```powershell
python src\backend\scripts\run_sample_parse.py
```

## Docker

Build and run the CPU API:

```powershell
docker build -f Dockerfile.backend -t paperlens-api .
docker run --env-file .env -p 8000:8080 paperlens-api
```

Build and run the frontend:

```powershell
docker build -f Dockerfile.frontend -t paperlens-ui .
docker run -e PAPERLENS_API_URL=http://host.docker.internal:8000 -p 3000:8080 paperlens-ui
```

For NVIDIA L4/CUDA deployments, use `Dockerfile.backend.gpu`.

## AWS deployment

Production runs in `ap-northeast-1` (Tokyo) and is defined in Terraform under
`infra/prod`:

- `https://paperlens.ailearn-vaic.cloud`: static UI on Vercel (project `paperlens`,
  `src/frontend/vercel.json`); Route 53 points the domain at Vercel
- `https://api.paperlens.ailearn-vaic.cloud`: API on ECS behind an Application Load
  Balancer. `compute_mode` in `infra/prod/prod.auto.tfvars` selects `cpu` (Fargate,
  8 vCPU / 32 GB, `Dockerfile.backend`, the default) or `gpu` (EC2 `g6.2xlarge`
  with an NVIDIA L4, `Dockerfile.backend.gpu`; requires G-instance quota)
- Paper artifacts in a private S3 bucket (IAM task role, no keys)
- `DATABASE_URL` and the LLM provider key in AWS Secrets Manager
- LLM and embeddings through OpenRouter's OpenAI-compatible API (`llm_base_url`,
  `llm_model` and `embedding_model` in `infra/prod/variables.tf`); if the
  embedding provider fails, retrieval falls back to BM25
- Supabase Postgres + Auth (Tokyo) for metadata and accounts

Infrastructure changes:

```bash
terraform -chdir=infra/prod plan
terraform -chdir=infra/prod apply
```

Runtime secrets (read from `.env`: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`,
`PROD_DATABASE_URL`; values are never printed):

```bash
python src/ops/put_aws_secrets.py
```

Release a new version from the repository root:

```bash
python src/ops/deploy_aws.py both          # or: api | ui
python src/ops/deploy_aws.py api --tag <existing-ecr-tag>
```

The deployed API image tag is stored in the SSM parameter
`/paperlens/prod/api-image-tag`, so a plain `terraform apply` never reverts the
running version. Non-secret Terraform inputs (alarm email, compute mode) go in
the gitignored `infra/prod/prod.auto.tfvars`; see `prod.auto.tfvars.example`.
The UI's runtime config (`PAPERLENS_API_URL`, `SUPABASE_AUTH_URL`,
`SUPABASE_PUBLISHABLE_KEY`) is set as Vercel project environment variables.

## Conversation context

The agent prompt is built in layers (`app/harness/memory.py`), each with a budget:
the fixed system prompt and tools; a memory block (structured state such as the
papers discussed, recent citations and the user's language, plus a rolling summary
of older turns); the most recent whole turns within `AGENT_CONTEXT_RECENT_TOKENS`,
with earlier tool results compacted to answer and citations; then the new message.
History is cut on turn boundaries, so a tool result is never separated from its
tool call, and the full history stays in the database (append-only). Summaries and
state help resolve references but are never cited. Follow-up and non-English
questions are rewritten into a standalone English retrieval query
(`app/rag/query_rewrite.py`). The `/papers/{id}/qa` chat stores its turns in the
same tables, scoped to the user.

## CI/CD

- `.github/workflows/ci.yml` runs on every pull request and branch push: backend
  tests and lint, frontend syntax checks, `terraform fmt` / `validate`.
- `.github/workflows/deploy.yml` runs on every push to `master`: CI, then the API
  (when `src/backend/app`, `pyproject.toml`, a backend Dockerfile or `infra/prod`
  changed) and the UI (when `src/frontend` changed). It calls the same
  `src/ops/deploy_aws.py` as a manual deploy, signs in to AWS through OIDC (no
  stored keys), refuses a release plan that changes anything except the API task
  definition and service, and ends with a `/health` smoke test. Infrastructure
  changes are still applied by hand with a reviewed `terraform apply`.

One-time setup, after `terraform apply` has created the deploy role:

| GitHub setting | Kind | Value |
|---|---|---|
| `AWS_DEPLOY_ROLE_ARN` | variable | `terraform -chdir=infra/prod output -raw github_deploy_role_arn` |
| `TF_VAR_ALARM_EMAIL` | secret | the `alarm_email` from your `prod.auto.tfvars` |
| `VERCEL_TOKEN` | secret | a Vercel access token |
| `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID` | variables | from `src/frontend/.vercel/project.json` |

## Observability

- **Logs:** JSON lines in CloudWatch (`/ecs/paperlens-prod/api`), each with the
  `request_id` also returned in the `X-Request-ID` response header. Saved Logs
  Insights queries: `paperlens/errors`, `paperlens/one-request`,
  `paperlens/latency-by-route`, `paperlens/llm-usage-by-model`.
- **Metrics:** CloudWatch namespace `PaperLens` (Embedded Metric Format, no
  agent): request latency and errors per route, chat time-to-first-byte, LLM
  calls/errors/latency/tokens per model, ingestion outcomes and duration,
  retrieval latency and evidence size, answer confidence, guardrail blocks.
- **Dashboard and alarms:** `terraform output dashboard_url`. Alarms (to
  `alarm_email`): ALB and app 5xx, unhealthy targets, CPU/memory, LLM errors,
  failed uploads, slow chat start, bursts of ERROR logs.
- **LLM traces:** LangSmith. Put the key in the `paperlens/prod/langsmith-api-key`
  secret, set `langsmith_enabled = true` in `infra/prod/release.auto.tfvars`, and
  apply. Traces include prompts and paper text, so they leave AWS.

## Core stack

- FastAPI and SQLAlchemy/PostgreSQL
- Docling PDF parsing
- LangChain chunking, embeddings, PGVector, BM25 and grounded QA
- LangGraph research orchestration and tool-calling agent
- OpenAI-compatible LLM/VLM providers
- AWS (ECS, ALB, S3, Secrets Manager, CodeBuild, CloudWatch) via Terraform; Vercel for the UI
- Static JavaScript frontend with streaming Markdown, KaTeX and SVG plots
