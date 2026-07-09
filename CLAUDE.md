# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目身份

本仓库是 **modelscope/sirchmunk**（无向量数据库的 Agentic Search 框架）的定制 fork，目标是**医保政策问答系统（MediPolicyAI）**。关键定制点集中在 `config/env.example`：

- `SIRCHMUNK_SEARCH_PATHS` 默认指向 `./policy-docs`（淄博/山东省医保政策文件，PDF/DOCX）
- `SIRCHMUNK_PUBLIC_SERVICE=true` —— 公共服务模式，仅暴露问答 API，隐藏监控/设置/上传/文件浏览/知识管理端点
- `CHAT_HISTORY_MAX_TURNS=0`、`SIRCHMUNK_RAG_TOP_K_FILES=3` —— 为政策问答优化，减少检索前的额外 LLM 调用

上游产品文档见 `README.md` / `README_zh.md`；面向 AI 代理的协作规范（结构、命令、提交、语言）见 `AGENTS.md`。两份文件均要求**中文**回复与提交信息，代码标识符/命令/路径保持原文。

## 常用命令

Python 包管理**必须用 `uv`**，不要用 pip/conda/poetry/python -m venv（AGENTS.md 明确约束）。

```fish
uv venv --python 3.12 .venv                  # 创建虚拟环境
uv pip install -e ".[all]"                     # 源码安装 + 全部可选依赖
source .venv/bin/activate.fish                 # fish 激活

sirchmunk init                                 # 初始化 ~/.sirchmunk/ 与 .env
sirchmunk serve --host 0.0.0.0 --port 8584     # 仅后端 API
sirchmunk web serve --dev                      # API(8584) + Next.js 热重载(8585)
sirchmunk search "问题" --mode DEEP            # CLI 搜索
sirchmunk compile --paths ./policy-docs        # （可选）预编译知识索引
```

前端（`web/`，Next.js 16 + React 19）：

```fish
cd web && npm install
cd web && npm run lint     # ESLint
cd web && npm run build    # 生产构建
```

测试：`pytest`（依赖 `requirements/tests.txt`），但**当前 `tests/` 目录为空** —— 新增测试时一并创建 `tests/<area>/test_*.py`。注意 `web/package.json` 的 `name` 是历史遗留值 `opentutor-web`，与项目实际名不一致。

格式化/Lint：pre-commit 配置 black + ruff，但**排除了 `scripts/` 和 `web/`**（见 `.pre-commit-config.yaml`）。

## 大局架构

### 双时间轴：Compile（离线）与 Search（运行时）

核心创新在 `src/sirchmunk/learnings/`（详见 `src/sirchmunk/learnings/README.md`）。理解这套双轨是理解整个系统的前提：

- **Compile-time**（`learnings/compiler.py` + `learnings/tree_indexer.py`）：`sirchmunk compile` 批量把文档编成层级树索引 + `KnowledgeCluster` + 跨引用图，产物缓存到 `{work_path}/.cache/compile/`。**可选**，但能显著提升大文档集精度。
- **Search-time**（`learnings/knowledge_base.py` + `learnings/evidence_processor.py`）：每次查询即时运行；当 compile 产物存在时**自动消费**（FAST/DEEP 的 Phase 0 embedding 复用；DEEP 的 tree 导航 + graph 一跳扩展），不存在时优雅降级为标准检索。

### 搜索流水线（`src/sirchmunk/search.py` 的 `AgenticSearch`）

无向量库（indexless），靠 ripgrep/ripgrep-all + LLM 推理。三种模式：

1. **FAST**（默认）：贪婪检索，约 2 次 LLM 调用，2–5s。
2. **DEEP**：Monte Carlo 证据采样 + ReAct 代理深度探索，10–30s。
3. **FILENAME_ONLY**：仅文件名匹配，**不需要 LLM**（因此也无需 API key）。

数据流：query → 关键词提取（多级粒度）→ `GrepRetriever`（`retrieve/text_retriever.py`，调 ripgrep-all）→ Monte Carlo 证据采样 → LLM 合成 ROI → 产出/复用 `KnowledgeCluster` → DuckDB + Parquet 持久化（`storage/knowledge_storage.py`）。复用判定：query embedding 与已有 cluster 余弦相似度 ≥ `CLUSTER_SIM_THRESHOLD`(默认 0.85) 即直接命中，省去整轮 LLM。

### 关键模块职责

| 路径 | 职责 |
|------|------|
| `src/sirchmunk/search.py` | `AgenticSearch` 编排器（`search`/`compile`/`compile_status`/`compile_lint` 统一入口） |
| `src/sirchmunk/learnings/` | compile + 运行时知识构建、tree 索引、Monte Carlo 采样、lint 健康检查 |
| `src/sirchmunk/retrieve/` | `GrepRetriever` indexless 检索 |
| `src/sirchmunk/scan/` | `DirectoryScanner`、文件/网页扫描 |
| `src/sirchmunk/agentic/` | `ReActSearchAgent`（DEEP 模式）及其工具集 |
| `src/sirchmunk/schema/` | 数据结构：`KnowledgeCluster`/`EvidenceUnit`/`SearchContext` 等 |
| `src/sirchmunk/storage/` | DuckDB + Parquet 知识持久化 |
| `src/sirchmunk/llm/` | `OpenAIChat`（统一 OpenAI 兼容接口，多 provider） |
| `src/sirchmunk/api/` | FastAPI 路由（search/chat/files/history/knowledge/monitor/security...） |
| `src/sirchmunk/cli/` | `sirchmunk` CLI 入口（`cli/cli.py`、`web_launcher.py`） |
| `src/sirchmunk_mcp/` | MCP 服务（stdio/http），供 Claude/Cursor 等接入 |
| `web/` | Next.js 前端（`app/` 路由、`components/`、`hooks/`、`lib/`） |

### 数据与配置落点

- 运行时数据在 `SIRCHMUNK_WORK_PATH`（默认 `~/.sirchmunk/`）：`.cache/history/`（DuckDB 会话历史）、`.cache/knowledge/`（Parquet cluster）、`.cache/compile/`（树索引 + manifest）。
- 真实 `.env` 放工作目录，**不**提交；`config/env.example` 是模板（`sirchmunk init` 会基于它生成）。
- API 默认端口 **8584**（Swagger `/docs` 仅在 `SIRCHMUNK_DEBUG=true` 时开放）；前端热重载 **8585**。

## graphify 知识图谱

项目在 `graphify-out/` 维护代码知识图谱。输入 `/graphify` 时先用已安装的 Graphify skill；处理代码库问题优先 `graphify query "<问题>"`（完整规范见 `AGENTS.md` 的 Graphify 章节）。`graphify-out/` 与 `policy-docs/` 已在 `.graphifyignore` 中排除，增量更新产生的未提交变更是正常现象。
