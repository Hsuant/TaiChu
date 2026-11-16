# 多轮 SFT 数据来源

本文记录用于补充 TaiChu SFT 的多轮对话数据。默认 profile
`80m_multiturn` 只启用许可清晰、能够直接转换为 `messages` 的来源；真实用户
日志和非商业许可来源单独列出，不进入默认训练集。

数据卡信息核验日期：2026-11-15。

## 默认来源

| 来源 | 语言 | 许可证 | 规模 | 当前采用内容 |
|---|---|---|---:|---|
| [opencsg/smoltalk-chinese](https://huggingface.co/datasets/opencsg/smoltalk-chinese) | 中文 | Apache-2.0 数据卡；同时受 OpenCSG 社区许可约束 | 70 万+ | 13 个多轮任务 Parquet 分片 |
| [shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) | 中英平行双语 | Apache-2.0 | 9 万+ | 中英 `common_*_70k` 对话 |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 英文 | MIT | 207,865 条 `train_sft` | 全部 `train_sft` Parquet |
| [OpenMOSS-Team/moss-002-sft-data](https://huggingface.co/datasets/OpenMOSS-Team/moss-002-sft-data) | 中文、英文 | CC-BY-4.0 | 约 116 万 | 中文 helpfulness 与 honesty |

SmolTalk-Chinese 的仓库元数据标记 Apache-2.0，但 README 要求商业使用向
OpenCSG 申请许可。正式商用前必须单独确认该许可证，不能只看仓库标签。

## 接入方式

下载脚本：

```powershell
D:\Environments\Anaconda\envs\AI\python.exe `
  .\scripts\download_multiturn_sft.py
```

默认写入：

```text
E:/Academic/Datasets/TaiChu/SFT/raw/
```

来源在 `configs/prepare_sft_data.yaml` 中通过两个通用适配器接入：

- `canonical_messages`：读取 Parquet、JSONL、JSON，并识别
  `messages`、`conversations`、`conversation` 以及 ShareGPT 的
  `human`/`assistant` 字段。
- `moss_sft`：解析 MOSS 的 `plain_text` 与 `<eoh>`/`<eoa>` 轮次标记。

新增来源统一要求至少两轮 user/assistant 对话，避免把单轮指令数据重复计入
多轮补充集。

## 暂不进入默认训练集

| 来源 | 原因 |
|---|---|
| [ServiceNow-AI/M2Lingual](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual) | 高质量多轮多语言，但许可证为 `CC-BY-NC-SA-4.0`，不适合默认混入可能商用的模型数据 |
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 真实用户日志，含人口属性、IP 与请求头风险，必须先做完整 PII/有害内容清洗 |
| [BelleGroup/multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) | GPL-3.0，且数据卡明确提示是未经严格校验的 ChatGPT 生成内容 |
| [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | MIT，但实际核对后当前 JSONL 中没有至少两轮 user/assistant 对话，不能计入多轮补充集 |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 人工生成与评分质量较高，但数据是分叉消息树，需要额外的质量路径选择器 |
| [OpenMOSS-Team/moss-003-sft-data](https://huggingface.co/datasets/OpenMOSS-Team/moss-003-sft-data) | CC-BY-4.0，可接入；当前先以 MOSS-002 和 ShareGPT 控制重复与下载体积 |

## 配比

`80m_multiturn` profile 在原有 129,000 条配额上增加 40,000 条：

- 中文通用：50,000 -> 70,000
- 英文通用：47,600 -> 67,600
- 总配额：169,000

数学、代码、科学和中文考试配额保持不变，避免新增对话数据挤占原有能力覆盖。

## 验证

```powershell
D:\Environments\Anaconda\envs\AI\python.exe -m pytest `
  tests\test_sft_multiturn_sources.py `
  tests\test_unified_offline_preparation.py -q
```

构建 Manifest 的 `multi_turn_samples` 与 `multi_turn_ratio` 字段用于确认补充
数据实际进入最终数据集，而不是只存在于 raw 目录。

当前下载子集的多轮记录数：

| 来源 | 可用多轮记录 |
|---|---:|
| SmolTalk-Chinese | 384,908 |
| ShareGPT 中文 | 54,490 |
| ShareGPT 英文 | 74,735 |
| UltraChat-200k | 207,853 |
| MOSS-002 中文 | 512,328 |
