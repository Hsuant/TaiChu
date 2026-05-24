<div align="center">

# TaiChu (太初)

> **从零实现 Decoder-Only Transformer，构建专属于自己的 LLM**  
> 涵盖 Tokenizer 训练、模型架构、预训练、Agent 扩展的全链路

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/Hsuant/TaiChu?style=social)](https://github.com/Hsuant/TaiChu/stargazers)

</div>

---

## 🎯 项目初衷

在 LLM 遍地开花的今天，TaiChu 选择了一条「反求诸己」的道路：

**从零开始，不依赖预训练模型，亲手写每一行代码。**

这不是一个简单的 API 调用封装，而是一个完整的 LLM 学习与构建工程。你会发现：

- **Tokenizer 怎么训？** —— 亲手实现 BPE 训练流程
- **Attention 怎么做？** —— 从 MHA 到 GQA 再到 Flash Attention
- **模型怎么训？** —— 从数据加载到分布式训练全链路打通
- **Agent怎么做** —— 预留扩展接口，深入探索

这里没有黑盒，每一层网络、每一个超参数，都清晰可见。

---

## ✨ 核心特性

| 模块 | 实现内容 | 亮点 |
|------|----------|------|
| **Tokenizer 训练** | BPE 分词器训练、词汇表构建、编解码 | HuggingFace `tokenizers` 高性能后端 |
| **模型架构** | Decoder-Only Transformer | MHA / GQA、RoPE、SwiGLU、RMSNorm |
| **训练流程** | 预训练 + 分布式训练 | 混合精度、梯度累积、Checkpoint 管理 |
| **模型规模** | 50M ~ 1B+ 可配置 | YAML 配置文件一键切换 |
| **Agent 扩展** | 预留 Tool Use / ReAct 接口 | Roadmap 规划中 |

---

## 📁 项目结构

```text
TaiChu/
├── configs/                     # 配置文件目录
│   ├── TaiChu_LLM.yaml         # 模型配置
│   └── tokenizer_config.yaml   # Tokenizer 训练配置
├── model/                       # 模型核心代码
│   ├── attention.py             # 因果自注意力（MHA / GQA + RoPE）
│   ├── transformer_block.py    # Transformer 解码器层
│   ├── model.py                 # TaiChuModel 主模型
│   ├── config.py                # 超参数配置类
│   ├── embedding.py             # 词嵌入层
│   ├── positional_encoding.py   # RoPE 旋转位置编码
│   ├── normalization.py         # RMSNorm
│   ├── output_head.py           # 输出头
│   └── moe.py                   # MoE（混合专家）扩展
├── tokenizer/                   # Tokenizer 训练与推理
│   ├── train_tokenizer.py       # BPE 分词器训练脚本
│   ├── tokenizer.py             # 分词器加载与使用
│   ├── decoder.py               # 推理解码
│   ├── corpus.py                # 训练语料处理
│   └── __init__.py
├── dataset/                     # 数据加载模块
│   ├── train/                   # 训练集
│   ├── eval/                    # 评估集
│   ├── test/                    # 测试集
│   └── llm_dataset.py           # 预训练数据集（HuggingFace 集成）
├── trainer/                     # 训练引擎
│   ├── distributed_train.py     # 分布式训练入口
│   ├── train_pretrain.py        # 预训练主逻辑
│   ├── train_tokenizer.py       # tokenizer训练主逻辑
│   └── __init__.py
├── utils/                       # 工具模块
│   ├── config_loader.py         # 统一配置加载
│   ├── checkpoint.py            # 检查点管理（保存/恢复）
│   ├── logger.py                # 日志管理
│   ├── model_utils.py           # 模型相关工具（模型权重初始化策略、模型摘要、参数与 FLOPs 统计、梯度监控等功能）
│   ├── train_utils.py           # 训练相关工具（随机种子设置、设备获取、时间格式化、参数量统计等）
│   ├── visualization.py         # 可视化相关工具
│   └── __init__.py
├── tests/                       # 单元测试
│   ├── test_model.py            # 模型前向/反向测试
│   └── test_tokenizer.py        # Tokenizer 测试
│   └── ……
├── README.md                    # 项目简介
├── requirements.txt             # 项目依赖
└── LICENSE                      # CC BY-NC 4.0 许可证
```

---

## 🚀 技术全景

### 🔤 Tokenizer：让模型「读懂」你的语言

```python
# 从零训练一个 BPE 分词器
from tokenizer import train_tokenizer

train_tokenizer(
    files=["./data/corpus.jsonl"],  # 训练语料
    vocab_size=50000,                # 词表大小
    output_dir="./my_tokenizer"      # 输出目录
)
```

**实现要点：**
- 基于 HuggingFace `tokenizers` 高性能后端
- 支持 BPE（Byte-Pair Encoding）算法
- 自定义特殊标记：`<|im_start|>`、`<|im_end|>`、`<|pad|>`
- 训练完成后导出 `tokenizer.json`，可直接用于模型编解码

### 🧠 模型架构：Decoder-Only 的核心实现

```python
# 加载配置并初始化模型
from utils.config_loader import load_model_config
from model.model import TaiChuModel

config = load_model_config("configs/TaiChu_77M.yaml")
model = TaiChuModel(config)
```

**架构亮点：**

| 组件 | 实现 | 说明 |
|------|------|------|
| **注意力机制** | MHA / GQA | 支持 Grouped Query Attention 减少 KV 缓存 |
| **位置编码** | RoPE（旋转位置编码）| 苏剑林团队论文实现，支持长序列外推 |
| **前馈网络** | SwiGLU | 现代 LLM 标配，比标准 FFN 更优 |
| **归一化** | RMSNorm | 比 LayerNorm 更快，效果相当 |
| **MoE（可选）** | 混合专家 | 支持动态路由扩展模型容量 |

### 📊 训练流程：从数据到 Checkpoint

```python
# 预训练启动命令（示例）
python -m trainer.train_pretrain --model_config configs/TaiChu_LLM.yaml --pretrain_config configs/train_pretrain_config.yaml
```

**训练特性：**
- ✅ 混合精度训练（FP16 / BF16）
- ✅ 梯度累积，突破显存限制
- ✅ 分布式训练（DDP），支持多卡扩展
- ✅ Warmup + Cosine Decay 学习率调度
- ✅ Checkpoint 滚动保存 + 最佳模型自动选取

**数据集处理：**
- 基于 HuggingFace `datasets` 库，支持 JSON/JSONL 格式
- 自动添加 BOS/EOS 标记，PAD 填充位置标签设为 -100 忽略损失

### 🔧 配置体系：模块化、可扩展

项目采用 **YAML 配置 + Python 数据类** 的双层设计：

- **模型配置**（`model/config.py`）：隐藏层维度、头数、层数、激活函数等
- **训练配置**（`configs/train_pretrain_config.yaml`）：学习率、batch size、优化器参数
- **Tokenizer 配置**（`configs/tokenizer_config.yaml`）：词表大小、训练语料路径

通过 `ConfigLoader` 统一加载，支持命令行参数覆盖。

---

## 📦 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+（推荐，CPU 也可运行但速度较慢，CPU仅用于测试）

### 安装依赖

```bash
git clone https://github.com/Hsuant/TaiChu.git
cd TaiChu
pip install -r requirements.txt
```

### 训练你的第一个模型

```bash
# 1. 准备语料（JSONL 格式，每条为 {"text": "内容"}）
# 2. 训练 Tokenizer
python -m trainer.train_tokenizer --config configs/tokenizer_config.yaml

# 3. 开始预训练
python -m trainer.train_pretrain.py --config configs/TaiChu_50m.yaml

# 4. 测试生成效果
python -c "from model.model import TaiChuModel; ..."  # 详见 tests/test_model.py
```

### 单元测试

```bash
python -m pytest tests/
```

---

## 🧪 模型测试

项目内置完整的单元测试，覆盖核心功能：

- 前向传播形状验证
- 带 labels 的损失计算
- 带 attention_mask 的因果掩码
- KV Cache 增量推理
- MoE 层路由测试
- ……


---

## 🗺️ 未来路线图

> 初心延续：从 Tokenizer → LLM → Agent 的完整链路

| 阶段 | 目标 | 状态 |
|------|------|------|
| **Phase 1** | Tokenizer 训练与模型基础架构 | ✅ 已完成 |
| **Phase 2** | 预训练全流程 + 分布式训练 | ✅ 已完成 |
| **Phase 3** | Instruction Tuning + SFT | 🔄 进行中 |
| **Phase 4** | Agent 能力扩展（Tool Use / ReAct / Memory）| 📅 规划中 |
| **Phase 5** | 模型部署与服务化（vLLM / TGI 集成）| 📅 规划中 |

**Agent 扩展方向：**
- 工具调用（Tool Use）：让 LLM 能够调用外部 API
- 记忆机制（Memory）：短期/长期记忆管理
- ReAct 框架：推理 + 行动的循环决策
- 多 Agent 协作与编排

---

## 🤝 贡献指南

欢迎共建！TaiChu 是一个完全开源的学习型项目，特别欢迎以下方向的贡献：

- 模型架构优化（如 MoE、MLA 等新机制）
- 更高效的训练 pipeline（数据集预处理加速）
- Agent 模块的代码实现
- Bug 修复与文档完善

**提交前请确保：**
1. 代码通过现有单元测试
2. 新增功能有对应的测试覆盖
3. 遵循 CC BY-NC 4.0 许可证规范

---

## 📜 许可证

本项目采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可证：

- ✅ 允许共享、复制、修改
- ❌ **禁止商业用途**（如需商业授权，请联系 hsuantx@163.com）
- 🔗 需标注原作者（ZhangYao Xiang）


---

## 📖 相关资源

- [博客主页](https://blog.devnest.top) —— 技术细节与学习笔记
- [Transformer 论文](https://arxiv.org/abs/1706.03762) —— Attention Is All You Need
- [RoPE 论文](https://arxiv.org/abs/2104.09864) —— 旋转位置编码
- [GQA 论文](https://arxiv.org/abs/2305.13245) —— Grouped Query Attention

---

## 🙏 致谢

- 感谢所有为开源 LLM 生态做出贡献的开发者们

---

<div align="center">

**💡 TaiChu 的意义：不只是一个模型，更是一份从零开始的 LLM 学习指南**

*如果这个项目对你有帮助，欢迎点亮 ⭐ Star*

</div>