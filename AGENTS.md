# Repository Guidelines

## 项目结构与模块组织

Sirchmunk 是一个 Python 包，并内置 Next.js Web UI。核心 Python 代码位于 `src/sirchmunk/`：API 路由在 `api/`，CLI 入口在 `cli/`，检索与搜索逻辑在 `retrieve/`，数据结构在 `schema/`。MCP 服务包位于 `src/sirchmunk_mcp/`。前端代码位于 `web/`，包括 `app/`、`components/`、`hooks/`、`lib/` 和 `types/`。配置示例在 `config/`，Docker 文件在 `docker/`，基准测试在 `benchmarks/`，OpenClaw 配方在 `recipes/`。

## 构建、测试与本地开发命令

- `uv venv --python 3.12 .venv`：使用 `uv` 创建本地虚拟环境。
- `uv pip install -e ".[all]"`：以源码模式安装，并包含所有可选依赖。
- `sirchmunk init`：创建本地工作目录，并基于 `config/env.example` 生成 `.env`。
- `sirchmunk serve`：启动后端 API 服务。
- `sirchmunk web serve --dev`：启动 API 与 Next.js 开发服务器。
- `cd web && npm install`：安装前端依赖。
- `cd web && npm run lint`：运行 Next.js/ESLint 检查。
- `cd web && npm run build`：构建 Web UI。
- `pytest`：运行 Python 测试。

Python 包管理和虚拟环境必须始终使用 `uv`，不要使用 `pip`、`python -m venv`、`virtualenv`、`poetry` 或 `conda` 管理本项目的 Python 依赖与环境。

## 编码风格与命名规范

Python 使用 3.10+，遵循现有风格：4 空格缩进，函数、变量和模块使用 `snake_case`，类使用 `PascalCase`，导入语句放在文件顶部。API 路由应按功能拆分在 `src/sirchmunk/api/`。TypeScript/React 遵循 `web/.prettierrc.json`：2 空格缩进、单引号、不使用分号、合法位置保留尾随逗号、行宽 100。React 组件使用 `PascalCase`，Hook 使用 `use...` 命名。

## 测试规范

Python 测试依赖位于 `requirements/tests.txt`，当前包含 `pytest` 和 `pytest-asyncio`。新增测试建议放在 `tests/` 下，文件命名使用 `test_*.py`，并尽量映射被修改的源码区域，例如 `tests/api/test_search.py`。Schema、storage、utils 变更优先补单元测试；API 或 CLI 行为变更应补集成测试。前端变更至少运行 `npm run lint` 和 `npm run build`。

## 提交与 Pull Request 规范

近期提交多使用简短祈使句，也可使用带范围的 Conventional Commit 风格，例如 `docs: 升级 MiniMax 默认模型到 M3`。提交信息、PR 标题、PR 描述、评审回复、变更说明和面向协作者的 Git 工具输出说明均使用中文。PR 应包含清晰描述、关联 issue、已执行的验证命令；涉及 UI 时附截图或录屏；涉及配置、迁移或依赖时明确说明影响。

## 协作与语言规范

本仓库协作默认使用中文。Agent 的所有回复、状态更新、代码审查意见、Git commit message、PR 标题和 PR 正文都应使用中文。代码标识符、命令、路径、环境变量和第三方 API 名称保持原文，例如 `SIRCHMUNK_SEARCH_PATHS`、`npm run build`。

## 安全与配置提示

不要提交密钥或本地生成状态。环境变量以 `config/env.example` 为模板，真实值保存在 Sirchmunk 工作目录中，通常是 `~/.sirchmunk/.env`。示例和日志中注意隐藏 `SIRCHMUNK_SEARCH_PATHS`、API key、Docker volume 路径等敏感信息。

## Graphify 图谱使用规范

本项目在 `graphify-out/` 下维护代码知识图谱，包含核心节点、社区结构和跨文件关系。

当用户输入 `/graphify` 时，必须先使用已安装的 Graphify skill 或以下规则，再执行其他代码库分析。

规则：
- 处理代码库问题时，如果 `graphify-out/graph.json` 存在，优先运行 `graphify query "<问题>"`。分析关系用 `graphify path "<A>" "<B>"`，解释单个概念用 `graphify explain "<概念>"`。
- hook 或增量更新后 `graphify-out/` 出现未提交变更是正常情况，不应因此跳过 Graphify。只有当任务本身涉及图谱过期、图谱错误，或用户明确要求不用 Graphify 时才跳过。
- 如果 `graphify-out/wiki/index.md` 存在，优先用它做宏观导航，而不是直接浏览全部源码。
- 只有在做整体架构审查，或 `query`、`path`、`explain` 信息不足时，才读取 `graphify-out/GRAPH_REPORT.md`。
- 修改代码后运行 `graphify update .`，保持图谱最新；该命令仅做 AST 更新，不消耗 API。
