# wechat-agent-lite

Lightweight daily article automation for WeChat Official Accounts.

微信公众号日更自动化系统，适合小规格服务器部署与日常运维。

## Overview

This project provides:

- daily health checks
- multi-source topic fetching
- source maintenance and repair
- topic ranking and article generation
- cover generation and validation
- WeChat draft publishing
- daily email reporting
- a web console for operations

## Language Versions

- English: [README.en.md](./README.en.md)
- 中文: [README.zh-CN.md](./README.zh-CN.md)

## Quick Start

```powershell
cd F:\projects\article_generation\wechat-agent-lite
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open:

- `http://127.0.0.1:8080`

## Repository Structure

- [app](./app)
- [config](./config)
- [deploy](./deploy)
- [docs](./docs)
- [tests](./tests)

## Deployment

Ubuntu deployment scripts:

- [deploy/bootstrap_ubuntu.sh](./deploy/bootstrap_ubuntu.sh)
- [deploy/deploy_uploaded_zip.sh](./deploy/deploy_uploaded_zip.sh)

Default deploy target:

- `/opt/wechat-agent-lite`

systemd service:

- `wechat-agent-lite.service`
