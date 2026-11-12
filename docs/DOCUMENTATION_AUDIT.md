# 文档审计与验证报告

> 审计日期：2026-11-12
> 审计对象：TaiChu 当前工作区、Git 历史、现有 Markdown、配置、代码路径与公开实验页面
> 审计边界：本轮只修改文档、文档导航和 `.gitignore` 中阻碍实验文档纳入版本控制的规则；
> 未修改模型、训练逻辑或实验配置。

## 1. 审计范围

### 仓库

- 审计当前工作区文件、`README.md`、`docs/`、`data/`、`dataset/`。
- 对照 `configs/`、`model/`、`training/`、`trainer/`、`eval/`、`chat/`。
- 检查 Git 提交历史、当前未提交状态和版本控制忽略规则。

### 外部参考

- MiniMind README：用于分析项目展示、徽章、模型表、快速开始和折叠式更新日志的
  组织方式，不复制其内容或数据。
- SwanLab `taichu_80m_4B`：读取运行状态、时间、环境、依赖和图表趋势。

## 2. 主要审计发现

| 等级 | 发现 | 处理 |
|---|---|---|
| P0 | README 引用已删除的审计、指标、模型产物、Tokenizer 报告和 `dataset/ARCHITECTURE.md` | 全部替换为当前有效文档 |
| P0 | README 章节顺序不符合项目展示要求，且混入大量配置手册内容 | 按 14 项要求重排，配置细节迁移到配置文档 |
| P0 | `data/README.md` 引用已删除的 `data/samples/` 和旧配置文档 | 重写为当前目录与命令 |
| P0 | 训练/数据文档测试数量不一致：243 与 249 | 统一为 249，并注明测试目录未纳入公开版本控制 |
| P0 | `.gitignore` 的 `experiments/` 同时忽略 `docs/experiments/` | 收窄为根目录 `/experiments/`，训练产物仍忽略 |
| P1 | 第一阶段实验没有统一档案和模板 | 新增实验索引、模板、80M 档案 |
| P1 | SwanLab 实际 LR 曲线约 `5e-4`，仓库默认配置为 `3e-4` | 明确区分“运行可见值”和“仓库默认值” |
| P1 | Binary Storage 文档未完整列出 `int16/uint16` | 按 `dataset/storage/binary.py` 修正 |
| P1 | 训练代码支持 GPU/AMP/NCCL 路径，但缺少目标硬件验证 | 在 README、状态和 Roadmap 中保留为待验证 |
| P1 | `tests/` 在当前 Git 历史被删除，工作区又忽略该目录 | 保留测试结果，但明确公开仓库不可直接复现 |
| P1 | `prepare_pretrain_data.yaml` 与 80M 训练配置的 Dataset ID 不一致 | 文档列为已知问题；按约束未修改配置 |
| P1 | Tokenizer 元数据许可证为 Apache 2.0，仓库为 CC BY-NC 4.0 | 文档明确发布前需统一元数据 |
| P2 | 旧文档包含“当前 103 GB 磁盘空间”等易过期状态 | 改为历史测量并提示重新评估 |

## 3. 文档结构

### README

最终顺序：

1. 项目首页
2. 项目介绍
3. 核心特性
4. 系统架构
5. 模型与 Tokenizer
6. 数据体系
7. 训练框架
8. 环境安装与快速开始
9. 训练实验
10. 模型评估与推理
11. 项目进度
12. 文档导航
13. 更新日志
14. Roadmap、贡献与许可证

更新日志使用 GitHub 原生 `details` / `summary` 标签，默认折叠。

### 新增维护体系

- `docs/README.md`
- `docs/project-status.md`
- `docs/roadmap.md`
- `CHANGELOG.md`
- `docs/experiments/README.md`
- `docs/experiments/TEMPLATE.md`
- `docs/experiments/2026-11-10-taichu-80m-4b.md`

### 新增技术文档

- `docs/MODEL_AND_TOKENIZER.md`
- `docs/CONFIGURATION.md`
- `docs/EVALUATION_AND_INFERENCE.md`

### 重写或修正

- `README.md`
- `data/README.md`
- `dataset/README.md`
- `dataset/offline/README.md`
- `docs/CHAT_ENTRY.md`
- `docs/PRETRAINING_DATA_PIPELINE.md`
- `docs/PRETRAIN_PROBE.md`
- `docs/TRAINING_ARCHITECTURE.md`
- `docs/UNIFIED_DATA_INFRASTRUCTURE.md`

## 4. 第一阶段实验核实

### 已核实

- 项目：`TaiChu`
- 工作区：`ByteWyrm`
- 运行名：`taichu_80m_4B`
- Run ID：`zxdscptz`
- 状态：完成
- 开始时间（SwanLab 页面）：2026-11-10 11:01:17
- 运行时长：22 小时 20 分 23 秒
- OS：Ubuntu 22.04.5 LTS
- Python：3.11.9
- PyTorch：2.9.0+das.opt1.dtk2604
- SwanLab：0.10.4
- 恢复来源：`checkpoint-26000.pt`
- 图表分区：`train`、`val`、`generation`

### 仅近似可见

- Train Loss 起始约 10–11。
- 约 5k Step 后 Train Loss 进入大约 3.5–4 的平台区。
- Learning Rate 峰值约 `5e-4`，末段衰减。
- X 轴延伸到约 30k Step。
- Throughput 常见约 2 万 Token/s。

近似读数未写成精确结论。

### 待补充

- 精确最终 Step 和消费 Token。
- 首末 Train Loss、Validation Loss/PPL/BPB/Accuracy。
- 精确 Learning Rate 和 Grad Norm。
- 实际 Dataset Manifest、来源占比与 Token 数。
- 加速器型号、数量、显存和利用率。
- Best/Final Artifact 路径。
- 图表截图或 CSV/JSON 导出。

## 5. 验证结果

### 自动化测试

执行：

```powershell
& 'D:\Environments\Anaconda\envs\AI\python.exe' -m pytest -q
```

结果：

```text
249 passed, 2 skipped, 150 subtests passed
17 warnings
54.21s
```

警告主要来自 PyTorch JIT 弃用提示和 MoE 小专家尺寸测试告警，未导致测试失败。

### Markdown

- 相对链接：全部解析到现有文件。
- README 页内锚点：全部可解析。
- 跨文档相对 Fragment：全部可解析。
- 代码围栏：成对闭合。
- `details` / `summary`：开始与结束标签数量一致。
- Markdown 行尾空白：0 处。
- 项目横幅：`docs/assets/taichu-banner.svg` 存在且 XML 可解析。
- 项目 Logo：README 使用作者提供的 `docs/assets/taichu-logo.svg`；
  banner 内嵌其圆形透明优化副本，避免 SVG-as-image 外部资源加载问题。
- `git diff --check`：无空白错误。

外部 HTTP 链接未执行批量 HEAD 校验。SwanLab 与 GitHub 在当前网络会话中出现连接
重置；SwanLab 页面已通过浏览器实际打开并读取，其他外部链接保留为用户可访问的
官方入口。

### Mermaid

当前文档包含 4 个 Mermaid 图：

- README 系统架构。
- README 训练流程。
- 模型与 Tokenizer 前向结构。
- 配置关系图。

已检查围栏、节点引号和流程图声明；本环境未安装 Mermaid CLI，未执行像素级渲染验证。
GitHub 渲染应在提交后用预览再次确认。

### 路径与命令

已检查以下关键路径存在：

- `model/model.py`
- `model/attention.py`
- `model/transformer_block.py`
- `model/feedforward.py`
- `training/model_info.py`
- `dataset/offline/sample_sources.py`
- `dataset/offline/prepare.py`
- `configs/TaiChu_80M.yaml`
- `configs/train_pretrain_80m.yaml`
- `configs/prepare_pretrain_data.yaml`
- `configs/sample_pretrain_80m.yaml`
- `taichu_tokenizer/tokenizer.json`
- `taichu_tokenizer/chat_template.jinja`

## 6. 未解决风险

1. **测试目录版本控制**：需要决定是否恢复 `tests/` 并接入 CI。
2. **实验配置身份**：`prepare_pretrain_data.yaml` 和训练配置的 Dataset ID、路径仍需统一。
3. **实验指标归档**：SwanLab 动态页面尚未导出精确数据。
4. **GPU/DCU 与多卡**：代码存在，但目标硬件稳定性尚未验证。
5. **许可证元数据**：Tokenizer 文件中的 Apache 2.0 字段与仓库许可证不一致。
6. **Mermaid 视觉验证**：当前只做静态检查，未执行本地渲染。
7. **外链可达性**：部分外站连接不稳定，未完成自动 HTTP 状态检查。

## 7. 维护验收

后续文档变更至少满足：

- 相对链接和页内锚点可解析。
- Mermaid、HTML 和代码围栏闭合。
- 路径、命令和配置字段来自当前仓库。
- 实验数字有来源，缺失项写“待补充”。
- 预训练基座、Chat 协议与 SFT/对齐能力不混写。
- 修改模型、数据或训练行为时同步更新对应文档和状态。
