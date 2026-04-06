# wechat-agent-lite

## 项目简介

`wechat-agent-lite` 是一个面向微信公众号内容生产场景的轻量级自动化系统，目标是在小规格服务器上稳定完成每天一次的内容采集、选题排序、文章生成、封面生成、草稿发布和日报通知。

项目同时提供了一个 Web 控制台，用来做配置管理、任务触发、运行排障、抓取源维护和运维观察。

## 最新版本说明

- 2026-04-06: [发布说明](./docs/release-notes-2026-04-06.md)

## 目标

- 每天先做一次健康检查，再做一次主流程任务
- 每次主流程只产出 1 篇文章草稿
- 把抓取、排序、写作、发布和运维信息都落到可观察的运行记录里
- 允许在任一步骤失败后进行人工干预
- 能在 2 核 2G 这类低资源机器上稳定运行

## 核心能力

- `health` / `main` 两类定时任务
- 支持 RSS、GitHub、HTML 列表页三类热点来源
- 抓取源健康状态维护、降权、停用和修复
- 规则打分 + 模型重排的选题链路
- 正文素材补全与事实压缩
- 文章生成与质量检查
- 封面提示词生成、出图与校验
- 微信公众号草稿发布
- 日报邮件发送
- Token / 延迟 / 存储占用统计
- 控制台中查看抓取源状态和维护进度
- 代理感知抓取与 Scrapling 回退
- 针对小服务器的有限并发抓取策略

## 技术架构

### 后端入口

- FastAPI 应用入口：[app/main.py](./app/main.py)
- Uvicorn 启动入口：[run.py](./run.py)
- API 路由：[app/api.py](./app/api.py)

### 工作流与调度

- 主编排器：[app/services/orchestrator.py](./app/services/orchestrator.py)
- 调度服务：[app/services/scheduler_service.py](./app/services/scheduler_service.py)

### 数据与配置

- ORM 模型：[app/models.py](./app/models.py)
- 数据库与 Session：[app/db.py](./app/db.py)
- 运行配置加载：[app/core/config.py](./app/core/config.py)
- 默认配置：[app/services/default_settings.py](./app/services/default_settings.py)

### 内容生产相关服务

- 抓取与正文提取：[app/services/fetch_service.py](./app/services/fetch_service.py)
- 抓取源维护：[app/services/source_maintenance_service.py](./app/services/source_maintenance_service.py)
- Scrapling 回退抓取：[app/services/scrapling_fallback_service.py](./app/services/scrapling_fallback_service.py)
- 模型网关：[app/services/llm_gateway.py](./app/services/llm_gateway.py)
- 写作模板：[app/services/writing_template_service.py](./app/services/writing_template_service.py)
- 微信发布：[app/services/wechat_service.py](./app/services/wechat_service.py)
- 邮件发送：[app/services/mail_service.py](./app/services/mail_service.py)

### 控制台前端

- 控制台模板：[app/templates/index.html](./app/templates/index.html)

## 主流程说明

### 健康检查任务

1. `HEALTH_CHECK`
2. 可选执行 `SOURCE_MAINTENANCE`

### 主任务

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

每一步都会写入运行记录，包括：

- 开始时间
- 结束时间
- 耗时
- 状态
- 错误信息
- 运行摘要
- 模型审计信息

## 抓取源维护

抓取源维护是当前项目里的重点子系统，它负责：

- 记录每个源的健康状态
- 探测 feed 是否可用
- 探测 HTML 列表页是否可作为回退来源
- 在必要时调用 Scrapling 获取候选文章列表
- 根据规则或 LLM 二审生成维护动作

支持的动作包括：

- `update_url`
- `switch_to_html_list`
- `lower_weight`
- `disable`
- `manual_review`

当前版本还支持：

- 在控制台展示每个源的健康状态
- 展示 `SOURCE_MAINTENANCE` 当前正在检查哪个源
- 只对低置信度或人工复核案例使用 LLM

## 模型角色划分

项目将模型职责拆开配置：

- `decision`
  - 选题判断
  - 事实压缩
  - 质量检查
  - 抓取源维护二审
- `rerank`
  - 候选内容重排
- `writer`
  - 正文写作
- `cover_prompt`
  - 封面提示词
- `cover_image`
  - 封面图片生成

每个角色都有独立的：

- provider
- base_url
- api_key
- model_id

## 并发策略

项目已经针对低资源服务器做了保守并发控制。

默认值：

- `fetch.concurrent_workers = 6`
- `fetch.per_host_limit = 1`
- `source_maintenance.inspect_workers = 3`
- `source_maintenance.per_host_limit = 1`
- `source_maintenance.scrapling_max_concurrency = 1`

设计原则是：

- 全局并发可以大于 1
- 单域名并发尽量限制为 1
- Scrapling 这种偏重型抓取路径单独限流
- 既保证吞吐，又避免在 2C2G 机器上打爆内存、触发反爬或造成 SQLite 写竞争

## 控制台模块

当前控制台包含：

- 任务触发
- Token 概览
- 存储概览
- 最近任务
- 抓取源状态
- 调度与质量配置
- 模型配置
- 微信公众号配置
- SMTP 与日报配置
- 代理配置
- 任务动作
- 任务详情

## 运行配置

运行期配置保存在 `config_entries` 表里，可以通过控制台动态修改。

常见配置包括：

- 调度时间
- 质量阈值
- 各角色模型设置
- 代理设置
- 微信设置
- SMTP 设置
- 抓取源维护参数
- 并发参数

敏感字段会加密存储在 SQLite 中。

## 数据模型

核心表包括：

- `config_entries`
- `runs`
- `run_steps`
- `llm_calls`
- `source_health_states`

其中：

- `runs` 记录整次任务
- `run_steps` 记录每一步
- `llm_calls` 记录真实模型调用
- `source_health_states` 记录抓取源的长期健康状态

## 本地开发

### Windows

```powershell
cd F:\projects\article_generation\wechat-agent-lite
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

打开：

- `http://127.0.0.1:8080`

### 主要依赖

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

## Ubuntu 部署

### 从解压目录部署

```bash
sudo bash deploy/bootstrap_ubuntu.sh /path/to/wechat-agent-lite
```

### 从压缩包部署

```bash
sudo bash deploy/deploy_uploaded_zip.sh /path/to/wechat-agent-lite-YYYYMMDD-HHMMSS.zip
```

### 服务信息

- systemd 服务名：`wechat-agent-lite.service`
- 目标部署目录：`/opt/wechat-agent-lite`

## SSH 隧道访问控制台

```bash
ssh -L 18080:127.0.0.1:8080 ShadowKun@<server-ip>
```

然后访问：

- `http://127.0.0.1:18080`

## 运维说明

- 如果模型未配置，部分链路会回退到 mock 或安全兜底逻辑
- 微信发布失败时，任务可以保持本地结果并标记为部分成功
- 代理分享链接可以解析为运行配置
- 抓取源状态和维护进度已经进入控制台
- 项目优先保证可观察性与运维透明度，而不是极限吞吐

## 测试

仓库内包含这些方向的单元测试：

- 抓取源维护
- LLM 网关超时策略
- 微信草稿重投
- 邮件服务
- 标题生成

示例：

```powershell
python -m unittest tests.test_source_maintenance_service tests.test_llm_gateway -v
```

## 相关文档

- 方案设计：[docs/wechat-agent-lite-solution-design.md](./docs/wechat-agent-lite-solution-design.md)
- 集成说明：[docs/INTEGRATIONS.md](./docs/INTEGRATIONS.md)
