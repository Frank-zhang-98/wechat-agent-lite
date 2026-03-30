# wechat-agent-lite

## Overview

`wechat-agent-lite` is a lightweight article automation system for WeChat Official Accounts. It is designed for a small Ubuntu server and focuses on daily operations: fetching trending topics, ranking candidates, generating one article, creating a cover, publishing a WeChat draft, and sending an operational report.

The project also includes a web console for configuration, job triggering, run inspection, source health tracking, and maintenance debugging.

## Goals

- Run a daily health check before the main workflow
- Generate one article draft per main run
- Keep source management, model routing, and publishing transparent
- Support manual intervention when a step fails
- Stay operable on low-resource infrastructure such as 2 CPU / 2 GB RAM

## Core Features

- Daily scheduler with separate `health` and `main` runs
- Multi-source collection from RSS, GitHub, and HTML list pages
- Source maintenance with source health state tracking, fallback discovery, and source repair actions
- Topic ranking pipeline with rule-based scoring and model-assisted reranking
- Source enrichment and fact compression before writing
- Article generation and quality checking
- Cover prompt generation, image generation, and validation
- WeChat draft publishing with partial-success fallback behavior
- Daily email report
- Token, latency, and storage metrics in the console
- Proxy-aware fetching and Scrapling fallback support
- Limited-concurrency fetching suitable for small servers

## Architecture

### Backend

- FastAPI app entry: [app/main.py](./app/main.py)
- HTTP server entry: [run.py](./run.py)
- API routes: [app/api.py](./app/api.py)

### Orchestration

- Main workflow and retry handling: [app/services/orchestrator.py](./app/services/orchestrator.py)
- Scheduler bootstrap: [app/services/scheduler_service.py](./app/services/scheduler_service.py)

### Data and Configuration

- SQLAlchemy models: [app/models.py](./app/models.py)
- Database/session setup: [app/db.py](./app/db.py)
- Runtime config loader: [app/core/config.py](./app/core/config.py)
- Default runtime settings: [app/services/default_settings.py](./app/services/default_settings.py)

### Content Pipeline Services

- Fetching and extraction: [app/services/fetch_service.py](./app/services/fetch_service.py)
- Source maintenance: [app/services/source_maintenance_service.py](./app/services/source_maintenance_service.py)
- Scrapling fallback: [app/services/scrapling_fallback_service.py](./app/services/scrapling_fallback_service.py)
- LLM gateway: [app/services/llm_gateway.py](./app/services/llm_gateway.py)
- Writing templates: [app/services/writing_template_service.py](./app/services/writing_template_service.py)
- WeChat publishing: [app/services/wechat_service.py](./app/services/wechat_service.py)
- Mail delivery: [app/services/mail_service.py](./app/services/mail_service.py)

### Frontend Console

- Jinja template: [app/templates/index.html](./app/templates/index.html)

## Workflow

### Health Run

1. Check proxy health
2. Optionally run source maintenance

### Main Run

1. `HEALTH_CHECK`
2. `SOURCE_MAINTENANCE`
3. `FETCH`
4. `DEDUP`
5. `RULE_SCORE`
6. `RERANK`
7. `SELECT`
8. `SOURCE_ENRICH`
9. `FACT_PACK`
10. `FACT_COMPRESS`
11. `WRITE`
12. `QUALITY_CHECK`
13. `COVER_5D`
14. `COVER_GEN`
15. `COVER_CHECK`
16. `WECHAT_DRAFT`

The run state, per-step details, model metadata, and summaries are persisted in SQLite for later inspection.

## Source Maintenance

Source maintenance is a first-class subsystem in this project.

It does the following:

- Tracks source health in `source_health_states`
- Probes feeds and HTML list pages
- Uses Scrapling as a fallback when plain feed probing is not enough
- Produces repair actions such as:
  - `update_url`
  - `switch_to_html_list`
  - `lower_weight`
  - `disable`
  - `manual_review`
- Shows current maintenance progress in the web console
- Exposes source health state through the API

The latest implementation also supports:

- Proxy forwarding into Scrapling
- Limited concurrency for low-resource servers
- LLM-assisted review only for manual-review or low-confidence source cases

## LLM Roles

The project separates model roles by responsibility:

- `decision`
  - selection
  - fact compression
  - quality check
  - source maintenance review
- `rerank`
  - reranking candidates
- `writer`
  - article generation
- `cover_prompt`
  - cover prompt planning
- `cover_image`
  - image generation

Each role has its own provider/base URL/API key/model ID settings.

## Concurrency Strategy

The current runtime is intentionally conservative for small servers.

Default limits:

- Fetch workers: `6`
- Fetch per-host limit: `1`
- Source maintenance workers: `3`
- Source maintenance per-host limit: `1`
- Scrapling max concurrency: `1`

This keeps network throughput reasonable while reducing the risk of memory spikes, database contention, or anti-bot throttling.

## Web Console

The console currently supports:

- Manual run triggering
- Token overview
- Storage overview
- Recent runs
- Source health
- Scheduling and quality settings
- Per-role model configuration
- WeChat settings
- SMTP settings
- Proxy settings
- Run actions
- Run detail inspection

## Configuration

Runtime settings are stored in `config_entries` and can be changed through the console.

Important groups include:

- Scheduler and quality
- LLM role configuration
- Proxy configuration
- WeChat credentials
- SMTP credentials
- Source maintenance limits
- Concurrency limits

Sensitive values are stored encrypted in SQLite.

## Data Model

Key tables:

- `config_entries`
- `runs`
- `run_steps`
- `llm_calls`
- `source_health_states`

This gives the project enough observability for debugging run history, model consumption, and source maintenance.

## Local Development

### Windows

```powershell
cd F:\projects\article_generation\wechat-agent-lite
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open:

- `http://127.0.0.1:8080`

### Main Dependencies

- FastAPI
- Uvicorn
- SQLAlchemy
- APScheduler
- requests
- PySocks
- feedparser
- PyYAML
- Jinja2
- cryptography
- scrapling
- curl_cffi
- playwright
- browserforge

## Ubuntu Deployment

### Bootstrap from an unpacked project

```bash
sudo bash deploy/bootstrap_ubuntu.sh /path/to/wechat-agent-lite
```

### Deploy from a packaged zip

```bash
sudo bash deploy/deploy_uploaded_zip.sh /path/to/wechat-agent-lite-YYYYMMDD-HHMMSS.zip
```

### Service

- systemd unit: `wechat-agent-lite.service`
- deploy target directory: `/opt/wechat-agent-lite`

## SSH Tunnel Access

```bash
ssh -L 18080:127.0.0.1:8080 ShadowKun@<server-ip>
```

Then open:

- `http://127.0.0.1:18080`

## Operational Notes

- Real model calls are optional; missing model configuration falls back to mock behavior where supported
- WeChat publish failure can still preserve local article results and mark the run as partial success
- Proxy share links can be parsed into runtime proxy settings
- Source maintenance and source health are now visible in the console
- The project is optimized for operational transparency rather than raw throughput

## Testing

The repository includes unit tests around:

- source maintenance
- LLM gateway timeout policy
- WeChat redraft flow
- mail service
- title generation

Example:

```powershell
python -m unittest tests.test_source_maintenance_service tests.test_llm_gateway -v
```

## Documents

- Solution design: [docs/wechat-agent-lite-solution-design.md](./docs/wechat-agent-lite-solution-design.md)
- Integration notes: [docs/INTEGRATIONS.md](./docs/INTEGRATIONS.md)
