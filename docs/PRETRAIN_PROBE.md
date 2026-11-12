# TaiChu 预训练续写测试

`python -m eval.pretrain_probe` 面向预训练检查点做原始文本续写测试。
它不使用 Chat Template，也不维护对话历史；每条输入都是独立前缀。

## 包结构

```text
eval/pretrain_probe/
├── __init__.py       # 公共 API
├── __main__.py       # python -m eval.pretrain_probe
├── cli.py            # 交互、批处理与输出
├── config.py         # 严格 YAML 配置
├── continuation.py   # 原始文本续写引擎
└── runtime.py        # 模型产物运行时适配
```

## 与评估程序的分工

| 程序 | 用途 |
|---|---|
| `eval/evaluate_pretrain_model.py` | 验证集 Loss、PPL、BPB、Accuracy 等定量指标 |
| `eval/pretrain_probe/` | 人工或批量观察 next-token 续写质量 |
| `python -m chat` | 按 Chat Template 进行多轮对话 |

## 启动

```powershell
python -m eval.pretrain_probe `
  --config configs/pretrain_probe_80m.yaml
```

单条续写：

```powershell
python -m eval.pretrain_probe `
  --config configs/pretrain_probe_80m.yaml `
  --prompt "清晨，阳光透过窗户洒进房间，" `
  --max-new-tokens 64
```

多条提示可使用重复的 `--prompt`，或读取文本/JSONL 文件：

```powershell
python -m eval.pretrain_probe `
  --config configs/pretrain_probe_80m.yaml `
  --prompts-file prompts.txt `
  --json
```

文本文件每条非空行是一条提示。JSONL 每行必须是包含非空 `prompt` 字段的对象。
`--json` 每条结果输出一个 JSON 对象，并关闭流式输出，便于后续统计和回归比较。

## 测试策略

- 同一组固定前缀和固定 seed 应产生可复现结果。
- 用 0 温度和贪心采样检查模型是否延续当前文本主题。
- 用默认采样检查多样性，但不要把随机样本当作能力结论。
- 同长度前缀下比较不同检查点，观察重复、停句和语义漂移。
- 将 JSONL 结果保存后，再结合验证集 PPL 决定检查点取舍。

实验 80M 权重是预训练基座。续写测试衡量的是 next-token 分布，不等价于指令遵循、
多轮对话或安全对齐评估。
