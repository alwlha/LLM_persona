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

使用 `uv` 或 `pip` 安装依赖：

```bash
uv pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录下创建 `.env` 文件（或修改已有的）：

```text
OPENAI_API_KEY=your_sk_key_here
```

框架会自动读取该 Key 用于闭源模型测试及 LLM-as-Judge 评分。

### 3. 配置模型与路径

修改 `config.yaml` 管理全局路径及 Base URL。
修改 `main.py` 中的 `API_MODELS_REGISTRY` 和 `LOCAL_MODELS_REGISTRY` 来注册新的模型简称。

---

## 📊 运行实验

### 基础测试 (BFI 问卷)

测试 GPT-5.2 在 BFI 问卷上的表现（遍历所有激活提示词）：

```bash
python main.py --models gpt-5.2 --tasks bfi
```

### 激发态对比测试

指定特定的激活提示词（如 Baseline 和 高外向激发）：

```bash
python main.py --models claude-4.5 --activations baseline high_extraversion --tasks bfi
```

### 开放生成任务 + 自动评分 (LLM-as-Judge)

运行社交场景生成任务，并让 GPT-5.2 对结果进行打分：

```bash
python main.py --models Llama-3-8B --tasks social_scenario --judge deepseek
```

### 全量测试

测试所有已配置的模型和任务：

```bash
python main.py --all --tasks bfi social_scenario
```

---

## 📁 项目结构

- `main.py`: 统一实验入口，解析命令行参数。
- `config.yaml`: 外部路径与 API 基础配置。
- `src/`:
  - `models/`: 模型驱动层（`api_model.py`, `local_model.py`）。
  - `tasks/`: 任务定义层（`bfi_task.py`, `generation_task.py`）。
  - `activation/`: 人格激发逻辑。
  - `scoring/`: 自动评分逻辑 (`LLMJudge`)。
  - `utils/`: 配置加载、日志与解析工具。
- `data/`:
  - `bfi.txt`: BFI-44 问卷题目。
  - `prompts.json`: 激活态系统提示词。
  - `tasks/`: 生成任务场景数据 (JSON)。
- `results/`: 存放实验结果。
  - `*_raw.json`: 单次实验的原始响应与得分详情。
  - `summary_results.csv`: 所有实验的维度得分汇总表。

---

## 🛠 扩展指南

1. **添加新任务**: 在 `data/tasks/` 下新建 JSON 文件，并在 `main.py` 中通过 `--tasks 文件名` 调用。
2. **添加新激活方法**: 修改 `data/prompts.json`。未来支持微调模型时，只需在 JSON 中扩展 `method` 字段。
3. **查看结果**: 运行结束后，打开 `results/summary_results.csv` 即可进行横向对比分析。
