# TaiChu 对话入口

`python -m chat` 是面向实验权重的交互式对话入口。它复用仓库中的
`TaiChuModel`、KV Cache、Tokenizer Chat Template 和检查点配置，
不引入第二套模型实现。

## 分层

| 模块 | 职责 |
|---|---|
| `chat/config.py` | 严格解析 YAML，校验采样参数和上下文预算 |
| `chat/runtime.py` | 读取检查点、构造模型、选择设备与推理精度 |
| `chat/tokenizer.py` | 封装 AutoTokenizer 与 Chat Template |
| `chat/session.py` | 维护多轮消息，按完整轮次裁剪上下文 |
| `chat/sampling.py` | Top-K、Top-P、温度、重复惩罚和可复现采样 |
| `chat/engine.py` | 流式 KV Cache 推理与会话事务提交 |
| `chat/cli.py` | 终端 REPL、单次请求和运行统计 |

推理引擎不依赖终端接口。后续接入 WebSocket 或 HTTP 时，可以直接复用
`ChatEngine.respond(..., on_text=...)`。

## 启动

通用启动方式：

```powershell
python -m chat --config configs/chat_80m.yaml
```

单次请求：

```powershell
python -m chat `
  --config configs/chat_80m.yaml `
  --prompt "用三句话解释什么是语言模型。" `
  --max-new-tokens 96
```

默认权重：

```text
experiments/80m/taichu_80m_pretrain_v1/best_models/best_model.pt
```

可通过 `--checkpoint` 切换到其他同架构的最小推理产物。模型结构直接读取产物
内嵌的 `model_config`，不再需要独立 `model_config` 配置。完整训练 Checkpoint
不能直接作为聊天权重，应先从 `checkpoints/` 使用评估或导出工具生成推理产物。

## 交互命令

| 命令 | 行为 |
|---|---|
| `/help` | 显示命令 |
| `/clear` | 清空会话历史 |
| `/system <文本>` | 更新系统提示并清空历史 |
| `/stats` | 显示上一轮 token 数、耗时和吞吐 |
| `/exit`、`/quit` | 退出 |

## 上下文策略

- `context_tokens: null` 时使用模型配置的 `max_position_embeddings`。
- 输入预算为 `context_tokens - max_new_tokens`。
- 超限时只删除最早的完整 user/assistant 轮次，始终保留系统提示和最新用户消息。
- 最新一轮单独超限时拒绝请求，避免截断用户输入造成静默语义变化。
- 生成失败或按 `Ctrl+C` 中断时，未完成的用户轮次会回滚。

## 验证

```powershell
python -m pytest -q tests/test_chat_entry.py
```

如果测试目录未纳入当前公开仓库，可先按“加载真实 Best Artifact”做手动 Smoke Test。

## 能力边界

实验 80M 检查点来自预训练，不是 SFT 或偏好对齐模型。入口程序提供正确的对话协议、
会话状态和推理工程，但不会改变基座模型的知识、指令遵循和安全对齐水平。
