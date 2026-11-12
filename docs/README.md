# TaiChu 文档中心

本目录维护 TaiChu 的架构、状态、配置、实验和维护规则。根目录
[`README.md`](../README.md) 负责项目展示，本页负责文档导航。

## 文档地图

| 类型 | 文档 | 解决的问题 |
|---|---|---|
| 当前事实 | [项目状态](project-status.md) | 现在完成了什么、正在做什么、还缺什么 |
| 未来计划 | [Roadmap](roadmap.md) | 后续阶段、优先级与验收标准 |
| 历史记录 | [CHANGELOG](../CHANGELOG.md) | 按版本记录 Added / Changed / Fixed / Removed |
| 实验档案 | [实验索引](experiments/README.md) | 每次训练运行的配置、数据、指标和结论 |
| 模型 | [模型与 Tokenizer](MODEL_AND_TOKENIZER.md) | 架构、参数量、BPE 协议与模板 |
| 配置 | [配置指南](CONFIGURATION.md) | 模型、训练、数据、SwanLab 的字段关系 |
| 数据 | [统一数据基础设施](UNIFIED_DATA_INFRASTRUCTURE.md) | Schema、Storage、Sampling、Packing、Runtime |
| 数据 | [预训练数据管线](PRETRAINING_DATA_PIPELINE.md) | 清洗、去重、滑窗、Manifest 与性能说明 |
| 训练 | [训练架构](TRAINING_ARCHITECTURE.md) | 组件分层、控制流、状态所有权与恢复 |
| 评估 | [评估与推理](EVALUATION_AND_INFERENCE.md) | 指标、独立评估、续写与 Chat |
| 推理 | [Chat 入口](CHAT_ENTRY.md) | 终端会话、上下文裁剪与采样 |
| 推理 | [预训练续写探针](PRETRAIN_PROBE.md) | 原始文本 Next-Token 观察 |
| 质量 | [文档审计报告](DOCUMENTATION_AUDIT.md) | 本轮审计、修复、验证与剩余风险 |

包级说明：

- [Data Directory](../data/README.md)
- [Dataset Package](../dataset/README.md)
- [Offline Preparation](../dataset/offline/README.md)

## 单一事实源

为避免 README、文档和配置各说各话，使用以下边界：

| 内容 | 单一事实源 |
|---|---|
| 项目定位与入口 | `README.md` |
| 当前完成度与阻塞项 | `docs/project-status.md` |
| 后续计划 | `docs/roadmap.md` |
| 历史变更 | `CHANGELOG.md` |
| 单次训练事实 | `docs/experiments/<run>.md` |
| 模型结构 | `configs/TaiChu_*.yaml` + `model/` + 架构测试 |
| 数据格式 | `dataset/schema/`、`dataset/storage/` + 数据测试 |
| 训练行为 | `training/`、`trainer/` + 训练测试 |

## 维护规则

1. **模型变化**：同步更新模型文档、参数量表、配置示例和架构测试说明。
2. **数据变化**：同步更新 Schema、Manifest、准备命令和项目状态。
3. **训练变化**：同步更新训练架构、Checkpoint、恢复流程和配置指南。
4. **新实验完成**：复制实验模板，补齐可核实指标，更新实验索引和项目状态。
5. **阶段完成**：同步更新 README、Roadmap、CHANGELOG 和项目状态。

## 事实分级

文档中的信息按以下等级标记：

- **已核实**：可由当前代码、配置、测试或公开实验页面直接确认。
- **待补充**：逻辑上应有，但当前工作区或公开页面无法核实。
- **未实现**：代码中不存在对应训练算法或评估能力，不得写成已支持。

外部链接只作为来源，不把未能读取的页面摘要当作项目事实。
