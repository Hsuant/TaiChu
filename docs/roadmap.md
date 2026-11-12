# Roadmap

本 Roadmap 只描述后续工作，不代表已完成能力。状态定义：

- `[x]` 已完成且有可核实证据
- `[ ]` 未完成
- `[~]` 部分完成，仍需补齐验收

## P0：实验可追溯与公开复现

### 1. 补齐第一阶段实验档案

- [~] 运行状态、开始时间、时长、Python/PyTorch 环境已核实。
- [ ] 从 SwanLab 导出 `train/loss`、`train/lr`、`train/grad_norm` 和验证指标。
- [ ] 记录最终优化 Step、实际消费 Token、首末 Loss、峰值 LR 与末段 LR。
- [ ] 归档该次运行的 `dataset_manifest.json`、Content Hash 和来源配比。
- [ ] 记录加速器型号、数量、显存、单卡/多卡模式。

**验收标准**：实验档案中的每个关键数字都有 SwanLab、Checkpoint、Manifest 或
日志来源；无法获得的字段明确写“待补充”。

### 2. 消除公开配置中的本地依赖

- [ ] 为 Tokenizer、抽样、准备和训练增加可提交的路径模板。
- [ ] 统一 `prepare_pretrain_data.yaml` 与训练配置中的 `dataset_id`。
- [ ] README 的快速开始命令可在全新克隆环境按步骤执行。
- [ ] 保存最小数据 Fixture 的公开生成方式，不提交生产语料。

**验收标准**：在没有作者本机路径的全新环境中，能够完成 Tokenizer 验证、
样例数据准备和 CPU 预检。

### 3. 测试与 CI

- [ ] 决定 `tests/` 的版本控制策略。
- [ ] 恢复公开可执行测试集。
- [ ] 增加 Markdown 链接、命令路径、Mermaid 围栏和 YAML 解析检查。
- [ ] 增加 CPU 最小训练和评估 Smoke Test。

**验收标准**：Pull Request 自动执行测试；文档链接或配置字段失效时 CI 失败。

## P1：训练可靠性

### 4. 单加速器实机审计

- [ ] 验证 CUDA 或目标加速器的 BF16/FP16 AMP。
- [ ] 对比 `torch.compile` 与 Eager 的吞吐、显存和数值稳定性。
- [ ] 验证滚动、最终与异常 Checkpoint。
- [ ] 从中断 Checkpoint 恢复并对照 Loss/DataState 连续性。
- [ ] 记录热身后吞吐、峰值显存和可复现命令。

**验收标准**：目标硬件完成至少一次短跑、一次中断恢复和一次独立评估。

### 5. 多卡与分布式

- [ ] 2 卡与 4 卡 NCCL 通信测试。
- [ ] DDP + Token 加权梯度累积数值一致性。
- [ ] 多卡 Checkpoint 只由主 Rank 写入且所有 Rank 同步。
- [ ] 单卡速度差异、NCCL Watchdog 和故障恢复演练。

**验收标准**：固定数据与种子下，多卡梯度、Loss 和恢复结果与约定容差一致。

### 6. 评估体系

- [ ] 固定验证集与评估脚本版本。
- [ ] 输出可归档的 JSON、日志和 Chart 数据。
- [ ] 区分 Base Model 与 Chat Model 的评估协议。
- [ ] 将生成样本与定量指标放在同一个实验档案中。

**验收标准**：同一 Checkpoint 可由独立命令复现指标和生成样本。

## P2：后训练能力

### 7. SFT 最小闭环

- [ ] 接入现有 `SFTSequenceSample` 与 Assistant-only Loss。
- [ ] 实现 SFT Trainer、Checkpoint、验证与推理脚本。
- [ ] 建立数据 Manifest、训练配置和独立实验档案。

**验收标准**：完成一次可恢复 SFT 运行，报告 Loss、验证指标和固定 Prompt 输出。

### 8. Preference / Reward

- [ ] DPO / IPO / KTO 的优化目标与测试。
- [ ] Reference Model 管理与显存策略。
- [ ] Reward Head、Pair Ranking Loss 和评估。

**验收标准**：Preference Pair 不拆分、Reference LogProb 可追踪、实验结果可复现。

### 9. RLHF / GRPO / Agent

- [ ] Rollout 环境、Experience Buffer 和策略更新。
- [ ] Reward、Advantage、Reference LogProb 与 Group 语义验证。
- [ ] Agent 工具调用协议、轨迹记录和安全边界。

**验收标准**：在线采样、训练、评估、失败恢复和审计日志形成闭环。

## P3：模型扩展与发布

- [ ] 150M / 190M 公开训练记录。
- [ ] 更长上下文与外推方案。
- [ ] 下游 Benchmark：C-Eval、CMMLU、OpenBookQA、PIQA 等。
- [ ] 模型转换、量化与第三方推理框架兼容。
- [ ] 正式版本标签、发布说明和数据/模型卡片。

**验收标准**：模型卡、配置、Tokenizer、评测命令、许可证和数据来源均完整。
