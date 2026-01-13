# 多模态摘要生成与奖励建模系统

Multimodal Summarization and Reward Modeling System - Week 8 Assignment

## 概述

本系统实现了完整的学术论文摘要生成与奖励建模流程，包括：
1. 使用本地 Ollama Qwen3:8b 模型生成摘要
2. 摘要对比与人工/自动标注
3. 基于 DeBERTa-v3 的奖励模型训练
4. 多维度评估（ROUGE、BERTScore、奖励分数）

## 功能特性

| 模块 | 描述 |
|------|------|
| **SummaryGenerator** | 通过 Ollama API 调用本地 Qwen3:8b 生成摘要，支持不同温度参数 |
| **AnnotationInterface** | 交互式摘要对比标注界面 |
| **RewardModelTrainer** | 基于 DeBERTa-v3-base 的奖励模型训练 |
| **SummaryEvaluator** | ROUGE、BERTScore、奖励模型评分 |
| **Pipeline** | 完整的端到端流程管理 |

## 环境要求

- Python 3.8+
- Ollama 服务（本地运行）
- 8GB+ RAM
- （可选）NVIDIA GPU 用于加速训练

## 安装步骤

### 1. 安装 Ollama

**Windows:**
```bash
# 从官网下载安装包
# https://ollama.com/download/windows
# 或使用 winget
winget install Ollama.Ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

### 2. 启动 Ollama 服务

```bash
ollama serve
```

### 3. 拉取 Qwen3:8b 模型

```bash
ollama pull qwen3:8b
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install torch transformers datasets evaluate trl
pip install bert-score rouge-score absl-py requests numpy
```

## 使用方法

### 快速开始

直接运行完整流程：

```bash
python multimodal_summary_reward.py
```

这将执行完整流程：
1. 生成 3 篇示例论文的摘要对
2. 自动标注（选择 temperature=0.3 的摘要）
3. 训练奖励模型
4. 评估模型性能

### 自定义使用

```python
from multimodal_summary_reward import Pipeline, SummaryGenerator

# 方式1：使用完整流程
pipeline = Pipeline()

# 步骤1：生成摘要对
papers = [
    {
        "id": "paper_001",
        "text": "论文全文...",
        "reference_summary": "参考摘要..."
    }
]
pipeline.step1_generate_summaries(papers, "output/summary_pairs.json")

# 步骤2：标注（自动模式）
pipeline.step2_annotate("output/summary_pairs.json", "output/reward_data.jsonl", auto_mode=True)

# 步骤3：训练奖励模型
pipeline.step3_train_reward_model("output/reward_data.jsonl", "output/reward_model")

# 步骤4：评估
results = pipeline.step4_evaluate(papers[:2], "output/reward_model")

# 方式2：单独使用摘要生成器
generator = SummaryGenerator("qwen3:8b")
summary = generator.generate_summary(paper_text, temperature=0.7)
summary_a, summary_b = generator.generate_summary_pair(paper_text)
```

## 项目结构

```
Homework8_1/
├── multimodal_summary_reward.py   # 主程序
├── requirements.txt                # 依赖列表
├── README.md                       # 本文档
├── summary_pairs.json              # 生成的摘要对（输出）
├── reward_data.jsonl               # 标注数据（输出）
└── reward_model/                   # 训练好的奖励模型（输出）
    ├── config.json
    ├── model.safetensors
    └── ...
```

## API 参数说明

### SummaryGenerator

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model_name` | str | "qwen3:8b" | Ollama 模型名称 |
| `temperature` | float | 0.7 | 采样温度（0.0-1.0） |
| `top_p` | float | 0.9 | Nucleus 采样参数 |
| `max_length` | int | 512 | 最大生成长度 |

### RewardModelTrainer

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model_name` | str | "microsoft/deberta-v3-base" | 基础模型 |
| `num_epochs` | int | 3 | 训练轮数 |
| `batch_size` | int | 8 | 批次大小 |
| `learning_rate` | float | 2e-5 | 学习率 |

## 输出文件说明

### summary_pairs.json
生成的摘要对，JSON 格式：
```json
[
  {
    "paper_id": "paper_001",
    "summary_a": "温度0.3生成的摘要...",
    "summary_b": "温度0.9生成的摘要..."
  }
]
```

### reward_data.jsonl
标注数据，JSONL 格式（每行一个 JSON）：
```jsonl
{"chosen": "优选摘要", "rejected": "拒绝摘要"}
```

### reward_model/
训练好的奖励模型目录，包含：
- `config.json` - 模型配置
- `model.safetensors` - 模型权重
- `tokenizer_config.json` - 分词器配置
- `vocab.txt` - 词汇表

## 评估指标

### ROUGE 分数
- **ROUGE-1**: 单词重叠度
- **ROUGE-2**: 双词组合重叠度
- **ROUGE-L**: 最长公共子序列

### BERTScore
- **Precision**: 精确度
- **Recall**: 召回率
- **F1**: F1 分数

### 奖励模型分数
- **平均分**: 摘要质量平均分
- **标准差**: 分数波动程度

## 常见问题

### Q: Ollama 连接失败
A: 确保 Ollama 服务正在运行：
```bash
ollama serve
curl http://localhost:11434/api/tags  # 检查服务
```

### Q: 编码错误（Windows）
A: 程序已内置 UTF-8 编码处理，如仍有问题：
```bash
set PYTHONIOENCODING=utf-8
python multimodal_summary_reward.py
```

### Q: 内存不足
A: 减小批次大小或使用更小的模型：
```python
pipeline.step3_train_reward_model(
    "reward_data.jsonl",
    "reward_model",
    batch_size=4  # 减小批次
)
```

### Q: 评估指标加载失败
A: 安装缺失的依赖：
```bash
pip install rouge-score bert-score
```

## 扩展与定制

### 使用其他 Ollama 模型
```python
generator = SummaryGenerator("llama3.2:3b")
generator = SummaryGenerator("qwen3-coder:14b")
```

### 自定义训练参数
```python
trainer = RewardModelTrainer("bert-base-uncased")
trainer.train(
    train_dataset=dataset,
    output_dir="my_model",
    num_epochs=5,
    batch_size=16,
    learning_rate=1e-5
)
```

### 交互式标注
```python
pipeline.step2_annotate(
    "summary_pairs.json",
    "reward_data.jsonl",
    auto_mode=False  # 启用交互模式
)
```

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline 流程                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐              │
│  │ 论文输入      │ ──>  │ Ollama API   │              │
│  └──────────────┘      │ Qwen3:8b     │              │
│                        └──────┬───────┘              │
│                               │                        │
│                               v                        │
│                        ┌──────────────┐              │
│                        │ 摘要对生成    │              │
│                        └──────┬───────┘              │
│                               │                        │
│                               v                        │
│                        ┌──────────────┐              │
│                        │ 人工/自动    │              │
│                        │ 标注         │              │
│                        └──────┬───────┘              │
│                               │                        │
│                               v                        │
│  ┌──────────────┐      ┌──────────────┐              │
│  │ JSONL 数据   │ <─── │ 标注数据      │              │
│  └──────┬───────┘      └──────────────┘              │
│         │                                              │
│         v                                              │
│  ┌──────────────┐      ┌──────────────┐              │
│  │ DeBERTa-v3   │ ──>  │ 奖励模型      │              │
│  │ 训练          │      │              │              │
│  └──────┬───────┘      └──────┬───────┘              │
│         │                      │                       │
│         v                      v                       │
│  ┌──────────────────────────────────────┐            │
│  │         评估 (ROUGE/BERTScore)       │            │
│  └──────────────────────────────────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 许可证

本项目为教学作业代码，仅供学习使用。

## 参考资料

- [Ollama 官方文档](https://ollama.com/docs)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [BERTScore 论文](https://arxiv.org/abs/1904.09675)
- [ROUGE 指标](https://en.wikipedia.org/wiki/ROUGE_(metric))

## 作者

Week 8 Assignment - Multimodal Summarization and Reward Modeling
