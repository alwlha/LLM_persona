# LLM Personality Experiment Framework (LLM-PEF)

这是一个用于研究大语言模型（LLM）人格表现的自动化实验框架。目前支持通过 **Prompt 激活**（激发态测试）在 **BFI-44 (Big Five Inventory)** 问卷及 **开放生成任务** 上测量模型的人格维度得分。

## 🌟 核心功能

- **多模型支持**：统一调用闭源 API 模型 (OpenAI 兼容格式) 与本地开源模型 (Transformers/PyTorch)。
- **人格激发 (Activation)**：支持加载多种系统提示词（Baseline/高外向性/高宜人性等）来测试模型的人格可塑性。
- **混合评估 (Evaluation)**：
  - **BFI-44**: 标准人格量表，采用规则计分逻辑。
  - **生成任务**: 社交场景回复任务，支持 **LLM-as-Judge** (由高能力模型打分)。
- **模块化设计**: 轻松扩展新的模型、激活方法（如微调/权重激活）或测试任务。

---

## 🚀 快速开始

### 1. 环境准备

推荐使用 `uv` 安装并同步依赖：

```bash
uv sync
```

如果你更习惯 `pip`，也可以：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录下复制并编辑环境变量文件：

```bash
cp .env.example .env
```

然后在 `.env` 中填入你的 Key：

```text
OPENAI_API_KEY=your_sk_key_here
```

框架会自动读取该 Key，用于闭源模型测试及 LLM-as-Judge 评分。

### 3. 配置模型与路径

修改 `config.yaml` 管理全局路径及 Base URL（激活配置目录为 `paths.activations_dir`）。
修改 `src/models/registry.py` 中的 `CLOSED_MODELS_REGISTRY` 和 `OPEN_MODELS_REGISTRY` 来注册模型简称。

---

## 📊 运行实验

### 基础测试 (BFI 问卷)

测试 GPT-5.2 在 BFI 问卷上的表现（闭源入口）：

```bash
python main_closed.py --model gpt-5.2 --activation-method prompt --activation-type base --task bfi
```

### 激发态对比测试

指定特定激活（如 prompt + extraversion）：

```bash
python main_closed.py --model claude-4.5 --activation-method prompt --activation-type extraversion --task bfi
```

### 向量激活测试（合并 assistant-axis 人格向量）

1. 先在 `assistant-axis` 中计算并导出 5 个人格向量（`openness` 等）。
2. 将导出的 `.pt` 文件复制到 `data/vectors/qwen3-8b/`。
3. 使用 `vector_extraversion` 激活项运行：

```bash
python main_open.py --model Qwen3-8B --activation-method vector --activation-type extraversion --task bfi
```

说明：该激活方式通过 `method: "vector"` 读取 `meta` 中的向量配置，并在本地模型推理时进行隐藏层激活注入。

如果想在一次运行中连续执行多个激活，可直接传多个 `--activation-type` 参数值；结果会写入同一个 `run_id` 目录并汇总到同一个 `summary_results.csv`：

```bash
python main_open.py --model Qwen3-8B --activation-method vector --activation-type base extraversion openness --task bfi
```

### 快速对比原始输出 vs 激活输出

可用下面脚本快速验证激活是否真的改变了回复（会同时打印并保存 JSON）：

```bash
uv run python scripts/compare_activation_outputs.py \
  --model-key Qwen3-8B \
  --prompt "请先做自我介绍，再给我一个周末学习计划。" \
  --activations base vector_openness
```

### 开放生成任务 + 自动评分 (LLM-as-Judge)

运行社交场景生成任务，并让 GPT-5.2 对结果进行打分：

```bash
python main_open.py --model Llama-3-8B --activation-method prompt --activation-type agreeableness --task social_scenario --judge deepseek
```

说明：`main_closed.py` 只支持 `prompt` 激活；`main_open.py` 支持 `prompt` 和 `vector`。
主流程会根据 `--activation-method` 自动读取 `data/activation/<method>.json`。

---

## 📁 项目结构

- `main_open.py`: 开源模型实验入口。
- `main_closed.py`: 闭源模型实验入口。
- `config.yaml`: 外部路径与 API 基础配置。
- `src/`:
  - `models/`: 模型驱动层（`api_model.py`, `local_model.py`）。
  - `tasks/`: 任务定义层（`bfi_task.py`, `generation_task.py`）。
  - `activation/`: 人格激发逻辑。
  - `scoring/`: 自动评分逻辑 (`LLMJudge`)。
  - `utils/`: 配置加载、日志与解析工具。
- `data/`:
  - `bfi.txt`: BFI-44 问卷题目。
  - `activation/prompt.json`: prompt 激活配置。
  - `activation/vector.json`: vector 激活配置。
  - `tasks/`: 生成任务场景数据 (JSON)。
- `results/`: 按模型和 run_id 分目录存放结果。
  - `results/<model>/<run_id>/raw/*.json`: 单次实验原始响应与得分详情。
  - `results/<model>/<run_id>/summary_results.csv`: 本次 run 的汇总表。

---

## 🛠 扩展指南

1. **添加新任务**: 在 `data/tasks/` 下新建 JSON 文件，并通过 `--task 文件名` 调用。
2. **添加新激活方法**: 修改 `data/activation/` 下对应 JSON。未来支持微调模型时，只需在 JSON 中扩展 `method` 字段。
3. **查看结果**: 运行结束后，打开 `results/<model>/<run_id>/summary_results.csv` 进行横向对比分析。
