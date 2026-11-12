# TaiChu 训练基础设施架构

本文档描述重构后的训练边界、核心接口和扩展方式。目标是把“模型定义”、
“数据生产”和“训练执行”分开，同时避免为尚未存在的训练任务提前引入复杂抽象。

## 1. 分层

```text
configs/ -> training/cli.py + training/config.py
                       |
                       v
          trainer/train_pretrain.py::run_pretraining
                |      |      |
                v      v      v
           model/   dataset/  training/*
                              |
                              v
          trainer/train_pretrain.py::CausalLMTrainer
                              |
                 +------------+------------+
                 v                         v
        CausalLMEvaluator           CheckpointManager
```

目录职责：

| 路径 | 职责 |
|---|---|
| `model/` | 模型结构、前向、Loss、生成和 KV Cache |
| `dataset/` | 类型化样本、离线准备、二进制存储、采样、Packing、Batch、Runtime |
| `training/config.py` | 训练配置数据类、严格 YAML 解析、跨模块校验 |
| `training/cli.py` | 参数定义和不可变配置覆盖 |
| `training/environment.py` | 设备、随机种子、torchrun/DDP 环境 |
| `training/optimization.py` | AdamW 参数分组、StepBudget、WSD 调度 |
| `training/model_setup.py` | `torch.compile`、编译预热、DDP 包装 |
| `training/builders.py` | 模型、Tokenizer、DataLoader 构建 |
| `training/checkpoint.py` | 原子保存、严格加载、RNG 与训练状态恢复 |
| `training/model_artifacts.py` | 最佳/最终最小推理产物的原子保存与严格加载 |
| `training/diagnostics.py` | 平滑 Loss、梯度范数稳定性和 MFU |
| `training/validation.py` | Token 级验证指标公式 |
| `training/evaluator.py` | 训练验证和独立评估共享的评估流程 |
| `training/trainer_types.py` | 训练器输入组件与运行参数的不可变契约 |
| `trainer/train_pretrain.py` | 预训练装配、恢复、主体训练循环、CLI 入口和执行生命周期 |
| `eval/evaluate_pretrain_model.py` | 独立评估 CLI，复用共享评估器 |

## 2. 核心接口

### 配置

`PretrainConfig` 聚合以下不可变配置：

- `OptimizerConfig`
- `SchedulerConfig`
- `UnifiedDataConfig`
- `TrainingConfig`
- `EvaluatingConfig`
- `EarlyStoppingConfig`
- `SwanLabConfig`

训练 batch 只有一个来源：`data.batching.micro_batch_size`。
`training.gradient_accumulation_steps` 只负责累积次数。未知配置键会直接报错，
避免拼写错误被默认值静默吞掉。

### 运行时

`RuntimeContext` 明确保存：

```text
rank, local_rank, world_size, device, backend
```

`RANK` 与 `LOCAL_RANK` 分离，因此多节点训练不会再把 local rank 误当作
global rank。单进程运行不会因为机器存在多张 GPU 而误启动 DDP。

### 训练

`CausalLMTrainer` 接收两个对象：

- `TrainerComponents`：模型、DataLoader、Optimizer、Scheduler、Evaluator、
  Checkpoint、日志等已构建组件。
- `TrainerSettings`：步数、间隔、AMP、设备和恢复位置等不可变运行参数。

训练器只负责训练执行，不负责读取 YAML、创建模型或决定实验目录。

### 评估

`CausalLMEvaluator.evaluate()` 返回 `EvaluationResult`。训练验证和独立评估
都使用该接口，避免两套循环在 `segment_ids`、`position_ids`、BPB、分位数和
分布式聚合上逐渐分叉。

### Checkpoint

`CheckpointManager.load()` 返回 `CheckpointLoadResult`：

```text
global_step, checkpoint
```

`checkpoints/` 只保存完整训练状态：

- 模型、Optimizer、Scheduler、GradScaler
- 模型配置
- 数据恢复位置
- 早停状态
- SwanLab 身份
- Python、NumPy、CPU Torch、CUDA RNG 状态

`checkpoint-final.pt` 使用固定文件名保存正常结束状态，不参与常规滚动删除。
`checkpoint-emergency.pt` 使用固定文件名保存最后一个异常状态。

### 推理产物

`BestModelStore` 和最终模型导出共用最小产物格式：

```text
best_models/best_model.pt
final_models/final_model.pt
```

只包含：

- `model_state_dict`
- `model_config`
- `global_step`
- `metrics`
- `artifact_variant`
- `stop_reason` 与创建时间

不包含 Optimizer、Scheduler、Scaler、RNG、DataState 或 SwanLab 状态。
两个文件都支持 `torch.load(..., weights_only=True, mmap=True)`。

## 3. 核心控制流

```text
parse CLI
  -> load strict YAML
  -> initialize runtime and seeds
  -> resolve experiment directory
  -> load tokenizer
  -> build model
  -> optional compile + shape warmup
  -> optional DDP
  -> build train/validation loaders
  -> build optimizer and WSD scheduler
  -> optional checkpoint resume
  -> build evaluator and diagnostics
  -> CausalLMTrainer.run()
       -> collect micro batches
       -> global token-weighted loss scaling
       -> backward under no_sync / autocast
       -> finite check across ranks
       -> clip, optimizer step, scheduler step
       -> periodic diagnostics
       -> checkpoint before validation
       -> shared evaluator
       -> best checkpoint, optional early stop
  -> checkpoints/checkpoint-final.pt
  -> final_models/final_model.pt
```

## 4. 状态所有权

| 状态 | 所有者 |
|---|---|
| 模型参数与 KV Cache | `model/` |
| 数据位置 `DataState` | `dataset.runtime` |
| 当前 step、最佳 Loss、早停 | `CausalLMTrainer` |
| 序列化 Checkpoint | `CheckpointManager` |
| 最佳/最终推理产物 | `BestModelStore` / `save_model_artifact()` |
| 训练指标 | `TrainingDiagnostics` |
| 验证结果 | `EvaluationResult` |
| 实验目录 | `ExperimentPaths` |
| 日志 sink | `LogManager` / `SwanLabLogger` |

## 5. 未来训练任务

可以复用的稳定能力：

- 配置解析、环境初始化、随机种子
- 数据 Schema、Binary Storage、Sampler、Packing、Batch Builder
- 模型构建、DDP、AMP、编译
- Optimizer、Scheduler、Checkpoint、RNG 恢复
- 日志、诊断、验证指标和独立评估框架

任务专属层应自行实现：

- SFT：复用 Causal LM objective，接入已有 loss mask。
- Preference / DPO：新增 pair 输入和 preference objective。
- Reward Model：新增 Reward Head 和 pair ranking objective。
- PPO / GRPO：新增 rollout、reference model、advantage 和策略目标。

不把所有训练任务强行压入一个万能 Trainer。共享组件保持稳定，任务差异由独立
objective、batch contract 和执行流程承载。

## 6. 验证

当前维护工作区完整测试：

```text
249 passed, 2 skipped, 150 subtests passed
```

CPU 模拟训练完成 79 步，训练、验证、Checkpoint 和独立评估均成功。
CUDA AMP、FlashAttention、NCCL 和多节点环境仍需目标硬件验证。

注意：当前公开 Git 历史已删除 `tests/`，而本地工作区将 `tests/` 加入忽略规则。
测试数字来自维护工作区；在公开仓库恢复测试目录前，该基线不能由新克隆直接复现。
