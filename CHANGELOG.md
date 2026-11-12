# Changelog

本文件记录用户可见的工程与文档变化。版本号从本轮文档基线开始建立；
历史提交没有 Git Tag，早期阶段以汇总方式记录，不补造版本号。

## [0.1.0] - 2026-11-12

首个文档化基线。该版本不代表正式模型发布，公开模型能力仍限定为
TaiChu-80M 预训练基座。

### Added

- 将项目作者提供的 TaiChu 太初 Logo 接入 README 与全流程生命周期横幅。
- 新增全流程生命周期横幅与统一文档视觉入口。
- 横幅明确区分 CURRENT、PLAN 与 CONTINUOUS，避免把后续训练能力误标为已实现。
- 新增模型与 Tokenizer、配置、评估与推理、文档审计文档。
- 新增 `docs/project-status.md`、`docs/roadmap.md` 和本 `CHANGELOG.md`。
- 新增 `docs/experiments/` 索引、模板和第一阶段 80M 实验档案。
- 新增文档维护规则、事实分级和实验记录规范。

### Changed

- README 按项目介绍、核心特性、系统架构、模型、数据、训练、快速开始、
  实验、评估、状态、文档、更新日志和后续计划的顺序重构。
- 统一测试基线为 `249 passed, 2 skipped, 150 subtests passed`。
- 统一 80M / 150M / 190M 参数量、配置路径、训练预算与产物说明。
- 数据文档中的历史性能数据改为“环境相关测量”，不再使用“当前磁盘空间”等易过期表述。

### Fixed

- 修复 README 对已删除文档、配置和数据样例目录的引用。
- 修复 `data/README.md` 中不存在的样例目录和文档链接。
- 修复训练与数据文档中不一致的测试数量。
- 修复 Dataset 文档对 Binary DType 范围的不完整描述。
- 修复实验能力描述，明确 SFT、DPO、RLHF 与 Agent 尚未实现。

### Removed

- 移除无法核实的实验指标、硬件能力和“已支持”暗示。
- 移除会随机器状态变化的磁盘空间判断。
- 移除 README 中重复且顺序错位的配置手册式内容。

## [Pre-0.1.0] - 2026-11-11

历史阶段汇总，不含正式版本号。

### Added

- 统一数据基础设施：Schema、Binary Storage、Sampler、Packing、Batch、Runtime Resume。
- 文档级预训练准备：精确/近重复去重、Cluster 隔离、滑窗与 Manifest。
- 独立 `training/` 组件层、最小推理产物、SwanLab 恢复和终端 Chat。

### Changed

- 预训练从单体训练脚本迁移为可复用组件与 `CausalLMTrainer`。
- Packed Sequence 使用 `segment_ids` 与块对角因果 Mask。
- 训练恢复同时保存模型、优化器、调度器、Scaler、RNG 和 DataState。

### Fixed

- 修复 80M 配置参数量，使架构与 80,757,248 参数一致。
- 修复验证指标的分块精确统计和有效 Token 加权。
- 修复 Packing 跨文档 Attention 与 Distributed Sampler 恢复问题。

## [2025-07-14] - Project Start

- 初始模型、KV Cache、预训练框架、日志和基础测试进入仓库。
- 后续演进覆盖 Attention、RoPE、FFN、Tokenizer、数据与训练基础设施。
