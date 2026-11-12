# 项目状态

> 更新时间：2026-11-12
> 维护原则：这里记录“当前事实”，未来计划统一写入 [Roadmap](roadmap.md)。

## 当前快照

| 维度 | 状态 |
|---|---|
| 当前阶段 | 预训练基座研究与工程验证 |
| 可用公开模型档位 | 80M / 150M / 190M 配置；完整权重不随仓库分发 |
| 第一公开实验 | TaiChu-80M，SwanLab Run `zxdscptz`，状态“完成” |
| 代码测试基线 | `249 passed, 2 skipped, 150 subtests passed` |
| 数据主路径 | 文档级抽样 → 去重/划分 → Binary mmap → Packing → Pretraining |
| 推理入口 | 独立评估、原始文本续写、终端 Chat |
| 指令模型 | 未实现 |
| SFT / DPO / RLHF / Agent | 训练算法未实现；仅部分数据 Schema 已存在 |

## 已完成

### Tokenizer

- Byte-level BPE、NFC、数字拆分和 9 个固定特殊 Token。
- 32,768 词表与 Hugging Face 兼容产物。
- Chat Template、特殊 Token 文件协议和 Tokenizer 验证入口。

### 模型

- Dense Decoder-only 模型结构。
- RoPE、RMSNorm、SwiGLU、MHA/GQA、KV Cache 和权重绑定。
- 80M / 150M / 190M 三档参数配置与架构测试。
- 可选 MoE 结构、路由辅助损失和独立 Loss 分量。
- Packed Sequence 的 `segment_ids` 块对角因果 Attention。

### 数据

- 多任务 Sample Schema 与严格校验。
- JSONL / JSONL.GZ / Parquet 读取。
- 文档级清洗、精确去重、MinHash/LSH 近重复检测。
- Cluster 级 train / validation / diagnostic 隔离。
- 2,046 Token 滑窗与 128 Token Context Overlap。
- `tcbin/tcidx`、Manifest、Content Hash、mmap Reader。
- Sampler、Mixer、Stream Packing、Batch Builder 与 Runtime Resume。

### 训练

- AdamW、WSD、AMP、梯度累积与裁剪。
- Token 加权 Loss、有限性检查和分布式同步保护。
- DDP、`torch.compile`、编译形状预热。
- 原子 Checkpoint、RNG/DataState 恢复、最佳与最终最小推理产物。
- TensorBoard、SwanLab、结构化日志和诊断指标。

### 实验与推理

- 80M 第一阶段公开预训练运行。
- 验证集定量评估与生成样本。
- 原始文本续写探针。
- 终端 Chat、上下文裁剪、流式输出和失败回滚。

## 进行中

| 工作项 | 当前状态 | 下一步 |
|---|---|---|
| 第一阶段实验补档 | 已有 SwanLab 运行元数据与曲线链接 | 补齐 Manifest、最终 Step、验证指标和加速器信息 |
| 配置可复现性 | 存在作者本机绝对路径与 Dataset ID 不一致 | 增加可提交的本地模板或覆盖机制 |
| 测试版本控制 | 当前工作区 `tests/` 被忽略，最近提交又删除该目录 | 决定是否恢复测试目录并接入 CI |
| GPU/加速器实机审计 | CPU 测试和 80M 公开实验已完成 | 记录目标设备、AMP、编译和吞吐 |
| 多卡验证 | 代码与测试路径存在 | 进行 2/4 卡 NCCL、恢复与故障演练 |

## 已知问题

| 优先级 | 问题 | 影响 | 处理边界 |
|---:|---|---|---|
| P0 | `tests/` 未纳入当前版本控制 | 公开仓库无法复现测试基线 | 需要单独决定测试目录发布策略 |
| P0 | `prepare_pretrain_data.yaml` 使用绝对路径，`dataset_id` 与训练配置不一致 | 默认命令不能直接复现实验 | 本轮不修改实验配置，配置指南明确说明 |
| P0 | 第一阶段精确指标尚未归档 | 不能量化最终模型质量 | 从 SwanLab 导出后更新实验档案 |
| P1 | GPU、DCU、AMP、多卡 NCCL 未完成统一实机审计 | 不能宣称通用硬件稳定性 | 在目标环境执行短跑与通信测试 |
| P1 | `tokenizer_config.json` 的 `model_info.license` 为 Apache 2.0 | 与仓库 CC BY-NC 4.0 冲突 | 正式发布前统一元数据 |
| P1 | 无下游 Benchmark | 无法评价知识、推理、代码与安全 | 固定评测版本和 Prompt 后再报告 |
| P2 | SFT/DPO/RLHF 只有部分数据契约 | 不能把 Schema 当作训练能力 | 任务算法与评估完成后再标记已实现 |

## 当前结论

TaiChu 已完成一条可审计的 **Tokenizer → 数据 → 预训练 → Checkpoint →
评估 → 推理** 工程链路，并完成一次 80M 公开预训练运行。项目尚未完成指令微调、
偏好优化、Agent 训练、GPU 多卡稳定性验证或下游能力评测。

本轮文档不改变模型、训练逻辑或实验配置。所有配置与训练代码仍以仓库当前实现为准。
