# <实验名称>

> 日期：YYYY-MM-DD
> 状态：计划中 / 运行中 / 完成 / 失败 / 中断
> 负责人：<姓名或账号>

## 1. 实验目标

说明本实验要验证的假设、与上一实验的差异，以及不包含的能力边界。

## 2. 运行信息

| 字段 | 值 |
|---|---|
| 平台 |  |
| 项目 / 工作区 |  |
| 运行名 |  |
| Run ID |  |
| 状态 |  |
| 开始时间 |  |
| 运行时长 |  |
| 代码版本 |  |
| 运行命令 |  |

## 3. 模型

| 字段 | 值 |
|---|---|
| 模型 |  |
| 参数量 |  |
| Hidden / Layers |  |
| Q / KV Heads |  |
| FFN / Activation |  |
| Context |  |
| Precision |  |

## 4. 数据

| 字段 | 值 |
|---|---|
| Dataset ID |  |
| Dataset Version |  |
| Manifest / Content Hash |  |
| 来源 |  |
| 原始文本量 |  |
| 唯一文档数 |  |
| 输入 Token |  |
| 监督 Token |  |
| Train / Validation / Diagnostic |  |

## 5. 训练配置

| 字段 | 值 |
|---|---|
| Optimizer |  |
| Learning Rate |  |
| Weight Decay |  |
| Scheduler |  |
| Warmup / Stable / Decay Steps |  |
| Micro Batch |  |
| Gradient Accumulation |  |
| World Size |  |
| Token / Step |  |
| Target Tokens |  |
| Final Step |  |
| Gradient Clipping |  |

## 6. 结果

### 训练曲线

| 指标 | 起始 | 最终 | 备注 |
|---|---:|---:|---|
| Train Loss |  |  |  |
| Validation Loss |  |  |  |
| Learning Rate |  |  |  |
| Grad Norm |  |  |  |
| Throughput |  |  |  |

### 验证指标

| 指标 | 值 |
|---|---:|
| `val_loss` |  |
| `val_perplexity` |  |
| `val_bpb` |  |
| `val_accuracy` |  |
| `val_ce_p50` |  |
| `val_ce_p90` |  |

### 生成观察

记录固定 Prompt、采样参数、输出原文和人工观察。不要把少量样本当作 Benchmark。

## 7. 产物

| 产物 | 路径 |
|---|---|
| Best Model |  |
| Final Model |  |
| Final Checkpoint |  |
| Evaluation JSON |  |
| SwanLab / TensorBoard |  |

## 8. 稳定性与异常

- 是否出现非有限 Loss / Gradient：
- 是否跳过更新：
- 是否发生通信或数据错误：
- 是否中断并恢复：

## 9. 结论

- 已验证：
- 未验证：
- 与假设是否一致：
- 下一步：

## 10. 复现命令

```powershell
# 写入实际可执行命令
```

## 11. 证据与待补充

- 原始日志：
- 图表导出：
- 数据 Manifest：
- 待补充字段及原因：
