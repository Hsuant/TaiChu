# TaiChu 系统审计与问题修复总报告

审计日期：2026-10-26  
审计范围：Tokenizer、数据管线、模型结构、损失与指标、训练/评估、配置、文档、测试  
验证环境：Python 3.13.0、PyTorch 2.10.0、CPU only、CUDA 不可用  
当前测试结果：`154 passed, 1 skipped, 120 subtests passed`

## 1. 当前结论

- Tokenizer 协议、模型词表、数据加载、模型损失、验证指标和模拟训练链路已完成一致性修复。
- 当前内置 Tokenizer 词表为 `32768`，特殊 Token ID 固定为 `0-8`。
- 主语言模型损失使用正确的 next-token shift，并忽略 `-100`。
- 验证损失按有效 Token 加权，新增 PPL、BPB、准确率、精确分位数和可选空前缀诊断。
- 无效指标统一为 `None/N/A/null`，不再产生 `Infinity/NaN`。
- 80M、150M、190M 三档模型配置及对应训练配置已完成并加入回归测试。
- 当前环境为 CPU，未验证 CUDA AMP、GPU FlashAttention 和多卡 NCCL。
- 当前模型仍缺少 activation checkpointing 和下游 benchmark 集成，属于后续工程项。

## 2. 系统状态

| 子系统 | 状态 | 当前依据 |
|---|---|---|
| Tokenizer | 通过 | 特殊 Token ID、文件协议、随机 Unicode 往返、batch/single 一致 |
| 数据加载 | 通过 | Full Load 与 Parquet 等价、Worker 分片、packing、labels 对齐 |
| 模型结构 | 通过 | Attention 参考公式、因果性、GQA/MHA、KV Cache、SwiGLU、RMSNorm |
| 模型损失 | 通过 | 手工 CE 对照、padding 屏蔽、无监督目标拒绝、MoE 辅助损失公式 |
| 验证指标 | 通过 | token 加权 loss、PPL、BPB、准确率、分位数、严格 JSON |
| 训练链路 | 通过 | 79 步 CPU 模拟训练、验证、checkpoint、独立评估 |
| GPU/DDP | 未验证 | 当前无 CUDA 环境 |
| 下游能力 | 未完成 | 尚无 MMLU/C-Eval/HumanEval/GSM8K 等集成 |

## 3. 问题总表

严重度定义：

- P0：会直接导致训练失败、监督错误或模型不可用。
- P1：会导致结果错误、显著资源浪费或关键路径不稳定。
- P2：会影响可维护性、可复现性、诊断或使用体验。
- P3：文档、兼容性或可选功能问题。

### 3.1 Tokenizer

| ID | 级别 | 问题 | 原因 | 修复 | 状态 |
|---|---|---|---|---|---|
| TKN-01 | P0 | 训练配置使用旧的特殊 Token 名称，与 Tokenizer 不匹配 | 特殊 Token 协议迁移后，配置、默认值、CLI 和测试语料未同步 | 全部统一为 `&lt;\|taichu_start\|&gt;/&lt;\|taichu_end\|&gt;` | 已修复 |
| TKN-02 | P0 | `tokenizer.json` 与 `tokenizer_config.json` 特殊 Token 不一致 | 旧产物与 Transformers 配置文件由不同流程生成 | 重建 9 个特殊 Token 的固定顺序和文件协议 | 已修复 |
| TKN-03 | P0 | 模型 `vocab_size=64001`，实际 Tokenizer 为 `32768` | 模型配置残留旧词表规模 | 三个样例模型和三个小模型预设统一为 `32768` | 已修复 |
| TKN-04 | P1 | 旧归一化会删除换行和缩进 | `\s+ -> " "` 与 `Strip` 改变了代码和空白语义 | 默认只使用 NFC，保留原始空白 | 已修复 |
| TKN-05 | P1 | `digit_split` 在新版 Tokenizers 上类型错误 | `SplitDelimiterBehavior` API 变化 | 优先使用兼容字符串参数，并保留旧版回退 | 已修复 |
| TKN-06 | P1 | Tokenizer 训练使用 Python 多进程导致内存放大 | 每个进程重复加载解释器和库 | 改为单进程流式读取，依赖 Tokenizers 原生并行 | 已修复 |
| TKN-07 | P2 | Tokenizer 训练缺少阶段资源数据 | 无 RSS、峰值、进度和阶段耗时记录 | 增加 ResourceMonitor 阶段报告 | 已修复 |

### 3.2 数据集与加载器

| ID | 级别 | 问题 | 原因 | 修复 | 状态 |
|---|---|---|---|---|---|
| DAT-01 | P0 | 数据侧提前移位 labels，模型内部再次移位 | 数据代码与模型损失职责重叠 | Collator 保持 labels 对齐，只由模型执行 shift | 已修复 |
| DAT-02 | P0 | PAD 与 EOS 使用同一名称 | 旧配置把 `<\|im_end\|>` 当作 PAD | 使用独立 `<\|pad\|>`，mask 为 0 时 label 为 `-100` | 已修复 |
| DAT-03 | P1 | 全量模式逐样本重复 Tokenize | 在 `__getitem__` 中执行编码 | 初始化时批量 Tokenize，之后读取紧凑 Arrow 表 | 已修复 |
| DAT-04 | P1 | Full Load 无输入内存上限 | 直接加载全部数据 | 增加文件大小预算、输出预算和告警阈值 | 已修复 |
| DAT-05 | P1 | Parquet 模式不是真正的有界流式读取 | 旧路径依赖整体 dataset 映射 | 改为 Row Group + RecordBatch、有界预取 | 已修复 |
| DAT-06 | P1 | 多 Worker 可能重复或遗漏样本 | 缺少 rank/worker 槽位所有权 | Row Group 确定性分配到 `rank × worker` | 已修复 |
| DAT-07 | P1 | 缺失文本列时静默填空字符串 | 旧代码使用 `sample.get(...)` | 显式列校验，默认 `error`，可选 `skip` | 已修复 |
| DAT-08 | P2 | 离线与在线数据契约不一致 | 两套 Tokenize/packing 路径独立实现 | 统一 TextTokenizer、Typed Sample、Binary Storage 和 Task Registry | 已修复 |
| DAT-09 | P2 | 特殊 Token 可能重复插入 | Tokenizer 自动特殊 Token 与 Packer 同时生效 | 编码固定 `add_special_tokens=False`，Packer 只插入一次 | 已修复 |
| DAT-10 | P2 | 测试语料仍含旧 `<\|im_start\|>/<\|im_end\|>` | 数据未随协议迁移 | 测试语料机械替换为当前 Token | 已修复 |
| DAT-11 | P2 | Packing 后注意力可跨越文档边界 | 当前只有 EOS，没有 document/segment ID | 保留为已知设计限制；建议后续增加块对角文档 mask | 待优化 |
| DAT-12 | P2 | 历史 Dataset 包装层和旧 Loader 重复 | 兼容代码长期保留 | 删除旧 Loader 与兼容配置，统一使用 Unified Binary Runtime | 已修复 |
| DAT-13 | P2 | 代码、样例语料和评估语料混在 `../dataset` | 代码包目录同时承担数据仓库职责 | 代码保留在 `../dataset`，数据迁到 `data/samples` 和 `data/eval` | 已修复 |
| DAT-14 | P2 | 预处理目录混合多个生命周期职责 | 打包、清洗和 Runtime 边界不清 | 离线入口统一到 `dataset.offline`，Runtime 统一到 `dataset.runtime` | 已修复 |

### 3.3 模型结构

| ID | 级别 | 问题 | 原因 | 修复 | 状态 |
|---|---|---|---|---|---|
| MOD-01 | P1 | SwiGLU 门控顺序错误 | 实现为 `gate * SiLU(up)` | 改为 `SiLU(gate) * up`，普通 FFN、共享专家和路由专家同步 | 已修复 |
| MOD-02 | P1 | `torch.compile` checkpoint 无法严格加载 | state_dict 残留 `_orig_mod.` | 保存前解包，加载时统一去前缀 | 已修复 |
| MOD-03 | P1 | 单进程可能误启动多卡 DDP | 旧逻辑按机器 GPU 数推断 world size | 仅 `WORLD_SIZE > 1` 时初始化 DDP | 已修复 |
| MOD-04 | P1 | padded batch 生成把 PAD 当作有效 Key | 缺少 2D attention mask 输入 | `generate` 支持并持续扩展 padding mask | 已修复 |
| MOD-05 | P1 | 生成超过上下文时静默丢弃历史 | 自动裁剪历史并重置缓存 | 生成前显式检查并抛错 | 已修复 |
| MOD-06 | P2 | `generate` 调用后永久切换 eval | 未恢复调用前 mode | `finally` 恢复训练/评估状态 | 已修复 |
| MOD-07 | P1 | 空 batch 没有可监督目标时可能产生 NaN loss | 未提前验证 shift 后有效 label | 模型前向前拒绝，报出明确错误 | 已修复 |
| MOD-08 | P1 | padding 位置保留有效 label 时可能静默训练错误目标 | 只信任调用方 | attention mask 为 0 的位置要求 label=`-100` | 已修复 |
| MOD-09 | P1 | GQA 复制后的缓存布局和增量掩码易错 | 多头复制、缓存长度与全局位置耦合 | 统一掩码计划，逐 Token/多 Token cache 测试 | 已修复 |

### 3.4 模型损失与 MoE

| ID | 级别 | 问题 | 原因 | 修复 | 状态 |
|---|---|---|---|---|---|
| LOSS-01 | P0 | 主损失 shift 难以验证 | 数据与模型都可能执行移位 | 明确模型唯一负责 `logits[:-1] / labels[1:]`，加入手工 CE 对照 | 已验证 |
| LOSS-02 | P1 | MoE router z-loss 使用 `mean(logits²)` | 与 ST-MoE 的 log-partition 正则不等价 | 改为 `mean(logsumexp(logits)²)` | 已修复 |
| LOSS-03 | P1 | top-k 负载均衡损失尺度不固定 | assignment fraction 未按 `top_k` 归一化 | 按 `num_tokens × top_k` 归一化 | 已修复 |
| LOSS-04 | P2 | MoE 共享专家数量可被静默忽略 | 实现只支持 0/1，配置未限制 | 配置明确拒绝大于 1 | 已修复 |
| LOSS-05 | P2 | 无监督目标时没有明确错误 | 交叉熵对全 `-100` 返回 NaN | 前向前检查并拒绝 | 已修复 |

### 3.5 指标与评估

| ID | 级别 | 问题 | 原因 | 修复 | 状态 |
|---|---|---|---|---|---|
| MET-01 | P1 | `loss >= 100` 被强制转换为 `inf` | 任意数值保护阈值 | 非有限才返回 `None`，不再使用 100 阈值 | 已修复 |
| MET-02 | P1 | 空白 PPL 在空字符串上返回 `inf` | 空序列没有 next-token 目标 | 改为可选显式 `[BOS, EOS]` 诊断 | 已修复 |
| MET-03 | P1 | 单 Token 空前缀会触发模型异常 | PPL 至少需要两个 Token | 不满足条件返回 `None` | 已修复 |
| MET-04 | P2 | 无有效 Token 时 accuracy 返回 `0.0` | 未定义与真实 0% 混淆 | 改为 `None` | 已修复 |
| MET-05 | P2 | 空分位数返回 `0.0` | 空集没有统计定义 | 改为 `None`，并使用精确线性分位数 | 已修复 |
| MET-06 | P2 | 训练器分位数使用固定 `[0,20]` 直方图 | 近似范围不可配置且与其他路径不一致 | 汇总逐 Token loss 后精确计算 | 已修复 |
| MET-07 | P1 | CPU/Gloo 分布式验证不聚合 | 只检查 `device.type == "cuda"` | 只要进程组初始化就 all-reduce | 已修复 |
| MET-08 | P1 | 评估 JSON 写入 `Infinity` | Python 默认 JSON 允许非标准数值 | `allow_nan=False`，无效值写 `null` | 已修复 |
| MET-09 | P1 | 缺少跨 Tokenizer 指标 | 只有 token-dependent PPL | 新增 `val_bpb`，排除特殊 Token | 已新增 |
| MET-10 | P2 | 吞吐按最大序列长度估算 | 动态 padding 下高估吞吐 | 改用实际 padded Token 数 | 已修复 |
| MET-11 | P2 | CPU 环境虚构 330 TFLOPS MFU | 未知设备使用 GPU 默认峰值 | 峰值未定义时禁用 MFU，显示 `N/A` | 已修复 |
| MET-12 | P2 | 名为 GNS，实际不是标准 GNS | 只有聚合梯度，没有 per-example 方差 | 改名为 `grad_norm_sq_variance_ratio` | 已修复 |
| MET-13 | P2 | 无梯度时 `grad_max/grad_min` 为 `±inf` | 未处理空梯度集合 | 返回 0 | 已修复 |

### 3.6 训练、日志与测试

| ID | 级别 | 问题 | 原因 | 修复 | 状态 |
|---|---|---|---|---|---|
| OPS-01 | P1 | 模拟训练因旧特殊 Token 在数据初始化阶段失败 | 特殊 Token 配置未同步 | 修正模型规模、训练配置和测试语料 | 已修复 |
| OPS-02 | P1 | Windows 控制台输出 `U+FFFD` 时 GBK 报错 | 标准 StreamHandler 不处理编码失败 | 增加 Unicode 安全 console handler | 已修复 |
| OPS-03 | P2 | 测试依赖未提交的 GPT-2 审计资产 | 可选基线资产不在仓库 | 缺失时显式跳过 2 项 | 已修复 |
| OPS-04 | P2 | 训练入口 `--help` 崩溃 | argparse 帮助文本中的 `%` 未转义 | 改为 `%%`，帮助命令已验证 | 已修复 |
| OPS-05 | P2 | 训练/评估指标实现重复 | 两条评估循环独立维护 | 新增共享 `compute_token_statistics` | 已修复 |
| OPS-06 | P3 | 文档测试数量、词表、特殊 Token 和命令过期 | 多次修复后未统一更新 | 本报告统一当前数字，README 重写 | 已修复 |
| OPS-07 | P3 | 依赖声明要求 PyTorch 2.11，实测环境为 2.10 | 依赖下限高于验证环境 | 调整为 `torch>=2.10.0` | 已修复 |
| OPS-08 | P3 | README 声称 Python 3.9，代码使用 `X \| Y` 类型语法 | 文档未匹配实际最低版本 | README 改为 Python 3.10+ | 已修复 |

## 4. 当前模型损失定义

主损失：

```text
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
loss = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)
```

约束：

- labels 必须与 input_ids 同形状、同设备且为 `torch.long`。
- attention mask 为 0 的位置必须使用 `labels=-100`。
- 有效 label 必须在 `[0, vocab_size)` 内。
- batch 至少包含一个可监督 next-token 位置。
- eval 模式下 MoE 辅助损失为 0，因此验证 CE 与语言模型 loss 可比。

MoE 总损失：

```text
total_loss = language_model_ce + sum(layer_auxiliary_loss)
```

## 5. 当前验证指标

| 指标 | 定义 | 适用范围 |
|---|---|---|
| `val_loss` | 有效 next-token 的 token 加权平均 CE | 主验证指标 |
| `val_perplexity` | `exp(val_loss)` | 同一 Tokenizer 内比较 |
| `val_bpb` | `content_nll / (ln2 × UTF-8 bytes)` | 跨 Tokenizer 比较 |
| `val_accuracy` | next-token top-1 正确率 | 辅助指标 |
| `val_ce_p50/p90/p95` | 逐 Token CE 分位数 | 误差分布 |
| `val_tokens` | 有效监督 Token 数 | 可复现性 |
| `val_target_bytes` | BPB 内容字节数 | 可复现性 |
| `empty_prefix_eos_ce/ppl` | 显式 BOS → EOS 转移 | 可选诊断，默认关闭 |

## 6. 当前小模型预设

| 规格 | 参数量 | 目标 Token | 推荐显存 |
|---|---:|---:|---:|
| 80M | 80,757,248 | 4.0B | 12-16GB |
| 150M | 149,057,280 | 5.2B | 24GB |
| 190M | 191,394,816 | 7.0B | 24GB 最低，40GB 推荐 |

配置、数据 GB 预算、batch 和准备命令见
[`../configs/SMALL_MODEL_CONFIGS.md`](../configs/SMALL_MODEL_CONFIGS.md)。

## 7. 验证证据

### 7.1 自动化测试

```text
154 passed, 1 skipped, 120 subtests passed
```

主要覆盖：

- Tokenizer 训练、保存、重载、特殊 Token 和 Unicode 往返
- Full Load 与 Parquet Sharded 等价性
- labels、attention mask、packing 和 Worker 分片
- Attention 手工参考、因果性、MHA/GQA、KV Cache
- SwiGLU、RMSNorm、MoE、损失、反向传播
- PPL、BPB、分位数、空前缀指标和严格 JSON
- checkpoint 保存、恢复、独立评估和生成

跳过的 1 项依赖仓库未包含的可选 GPT-2 审计资产。

### 7.2 模拟训练

CPU 模拟训练共 79 个优化步：

- 验证 loss：`9.8582 → 7.1582`
- 验证 BPB：`3.1519 → 2.3135`
- 空前缀 EOS PPL：`13712.93 → 691.27`
- 训练、验证、checkpoint 和日志均正常

独立加载最终 checkpoint：

- `val_loss = 7.1518`
- `val_perplexity = 1276.43`
- `val_bpb = 2.3115`
- `val_accuracy = 0.1979`
- 空前缀 EOS PPL：`688.92`

这些结果只用于验证训练和评估链路，不代表 79 步模型的生成质量。

## 8. 剩余风险与后续工作

| 项目 | 影响 | 建议 |
|---|---|---|
| CUDA/AMP/FlashAttention 未验证 | 不能宣称 GPU 路径已通过 | 在 Ampere/Hopper GPU 复跑完整测试和单步训练 |
| 多卡 DDP/NCCL 未验证 | 不能宣称多卡稳定性 | 使用 `torchrun` 做 2/4 卡短跑 |
| 未实现 activation checkpointing | 12-16GB 训练 150M/190M 受限 | 在 TransformerBlock 前向中增加可配置 checkpoint |
| 无下游 benchmark | 无法评价知识、推理、代码与安全 | 接入 MMLU、C-Eval、HumanEval、GSM8K 等 |
| Packing 可跨文档注意力 | 可能混合无关文档 | 增加 document ID 和块对角 attention mask |
| 分布式分位数全量 gather | 大验证集会增加 CPU 内存 | 增加分块量化或可配置采样 |
| MFU 为近似公式 | 不能作为精确硬件效率 | 使用 FLOP 计数工具并含 attention/通信 |
| 生成采样仍在 Model 内部 | 策略扩展性一般 | 抽象 Sampler 接口 |
| Repetition penalty/beam search 未实现 | 生成策略有限 | 作为可选扩展，不计为当前缺陷 |

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| [`../README.md`](../README.md) | 当前入口、快速开始、模型与文档导航 |
| `AUDIT_REPORT.md` | 本文档，总体问题、原因、修复和风险 |
| [`../dataset/DATASET_PIPELINE_REPORT.md`](../dataset/DATASET_PIPELINE_REPORT.md) | 数据加载器设计与基准附录 |
| [`TOKENIZER_OPTIMIZATION_REPORT.md`](TOKENIZER_OPTIMIZATION_REPORT.md) | Tokenizer 训练与性能专项附录 |
| [`METRICS_AUDIT.md`](METRICS_AUDIT.md) | 指标和损失专项附录 |
| [`../configs/SMALL_MODEL_CONFIGS.md`](../configs/SMALL_MODEL_CONFIGS.md) | 80M/150M/190M 配置与数据预算 |
| [`UNIFIED_DATA_INFRASTRUCTURE.md`](./UNIFIED_DATA_INFRASTRUCTURE.md) | 全生命周期数据架构、API 与约束 |
| [`../dataset/README.md`](../dataset/README.md) | Dataset 包目录和职责边界 |
| [`../data/README.md`](../data/README.md) | 样例数据和生产数据目录 |

## 10. 复现命令

```powershell
$PY = "D:\Environments\Anaconda\envs\AI\python.exe"

& $PY -m pytest -q
& $PY -m trainer.train_pretrain --help
& $PY -m trainer.validate_tokenizer `
  --config configs/TaiChu_Tokenizer_config.yaml `
  --tokenizer_dir taichu_tokenizer `
  --vocab_size 32768
```

模拟训练：

```powershell
& $PY tests/test_train/run.py
```

小模型训练示例：

```powershell
& $PY -m trainer.train_pretrain `
  --model_config configs/TaiChu_Small_150M.yaml `
  --pretrain_config configs/train_small_150m.yaml
```
