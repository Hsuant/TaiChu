# TaiChu Byte-Level BPE Tokenizer 专项报告

> 本文是 [`AUDIT_REPORT.md`](AUDIT_REPORT.md) 的 Tokenizer 附录。文中的旧实现
> 缺陷已修复；当前状态和总体结论以审计总报告为准。

## 1. 环境与结论摘要

验证环境：

- Windows 11
- Python 3.13.0 (`D:\Environments\Anaconda\envs\AI\python.exe`)
- `tokenizers==0.22.2`
- `transformers==5.9.0`
- `psutil==7.0.0`
- 14 个物理核、20 个逻辑核

已确认的主要问题是：

1. 旧语料读取默认把 `num_workers=0` 解释为 20 个 Python 进程，配置又固定为 8 个进程；每个进程重复驻留解释器和库，导致进程树 RSS 和耗时同时恶化。
2. 当前配置的 `digit_split: true` 在 `tokenizers==0.22.2` 下直接抛出 API 类型错误，旧训练流程无法完成。
3. 旧特殊 Token、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json` 和 Chat Template 使用了不同协议。
4. 旧保存流程先保存一次 `tokenizer.json`，再通过 Transformers 保存整套文件，并手工重写配置；这会增加峰值内存，且手写配置与 Transformers 5.x 不兼容。
5. 旧归一化中的 `\s+ -> " "` 和 `strip` 会破坏换行、制表符和代码缩进，导致编码解码不可逆。
6. 旧流程没有阶段 RSS、峰值、进度和吞吐日志，无法区分数据驻留、进程放大和 BPE 原生工作集。

优化后，在同一 5,731,837 字符基准语料上：

- 端到端峰值进程树 RSS：`253.3 MB -> 58.14 MB`，下降 `77.0%`
- 端到端耗时：`5.191 s -> 0.795 s`，下降 `84.7%`
- 资源受限模式：峰值 `38.72 MB`，耗时 `2.157 s`
- 固定质量样例总 Token 数：`216 -> 220`（`+1.85%`）
- 可逆样例：`5/7 -> 7/7`；增加的 Token 来自保留换行、制表符和缩进

## 2. 现有代码分析

### 2.1 旧执行路径

真实调用链为：

`trainer/train_tokenizer.py::main`

-> 加载 YAML

-> `tokenizer.tokenizer.TaiChuTokenizerTrainer`

-> `tokenizer.corpus.CorpusIterator`

-> `Tokenizer.train_from_iterator`

-> `Tokenizer.save`

-> `PreTrainedTokenizerFast.save_pretrained`

-> 手工覆盖 `tokenizer_config.json` / `special_tokens_map.json`

旧实现并没有一次性读取整个 JSONL：

- 单进程路径逐行 `json.loads` 后立即 `yield`。
- 多进程路径只缓存一个 `chunk_size` 行块。
- 训练 API 使用 `train_from_iterator`，不是全量 `list`。

因此，“完整 JSONL 被无条件加载到内存”不是本次已确认的根因。问题集中在多进程放大、原生 BPE 工作集、重复保存和配置错误。

### 2.2 已确认缺陷

| 严重度 | 旧实现位置 | 问题与机制 | 处理 |
|---|---|---|---|
| P0 | `tokenizer/corpus.py:115`、`:184-197`（HEAD） | `num_workers=0` 展开为全部 20 核；`Pool` 每个子进程重复导入 Python 与库。主进程 RSS 看不到这些开销，必须统计进程树。 | 移除训练热路径 Python 进程池，默认单进程读取，Tokenizers 原生并行负责预分词。 |
| P0 | `tokenizer/pre_tokenizer.py:50-55`（HEAD） | `SplitDelimiterBehavior.REMOVED` 在 `tokenizers 0.22.2` 中被要求传 `"removed"`，训练启动即失败。 | 新 API 优先传字符串，并兼容旧版本枚举。 |
| P0 | `tokenizer/tokenizer.py:169-172`（HEAD） | 训练配置使用 `<\|im_start\|>/<\|im_end\|>`，磁盘配置声明 `<\|taichu_start\|>/<\|taichu_end\|>/<\|pad\|>`；实际 `tokenizer.json` 又是旧 added tokens。 | 以现有配置声明的 9 个 Token 名称和 ID 顺序重建协议。 |
| P1 | `tokenizer/tokenizer.py:155-161`、`:201-209`（HEAD） | 同一 Tokenizer 先直接保存，再由 `PreTrainedTokenizerFast` 保存一次；Transformers 包装和序列化同时增加 RSS。 | 原生 `tokenizer.save` 一次落盘；完整 Transformers 验证拆为独立命令。 |
| P1 | `configs/TaiChu_Tokenizer_config.yaml:34-41`（HEAD） | `Replace(r"\s+", " ")` 与 `Strip` 改变训练文本语义，吞掉代码换行和缩进。 | 默认只保留 NFC；不增加无必要清洗。 |
| P1 | `tokenizer/tokenizer.py:212-234`（HEAD） | 手工构造不完整的 `tokenizer_config.json`；在 Transformers 5.9 下 `additional_special_tokens` 格式不可重载。 | 显式写标准 Token、additional Token、`added_tokens_decoder`、Chat Template 和词表长度。 |
| P2 | `tokenizer/corpus.py:54-61`（HEAD） | 无效 JSON、缺字段、空文本全部静默跳过，没有诊断计数。 | 有界统计无效记录、缺字段、空文本和截断数量，不保存原始记录。 |
| P2 | 旧训练入口 | 没有阶段 RSS、峰值进程树、进度和吞吐。 | 新增 `ResourceMonitor` 阶段报告。 |

### 2.3 已排除或未证实的事项

- 未发现完整 JSONL 被转为 `list` 的代码路径。
- 未发现训练阶段无界 Python 缓存；旧多进程只保留一个有界行块。
- 未证明 Tokenizers 原生代码存在泄漏。关闭原生并行后峰值仍有约 230 MB，说明主要是合理工作集，而非单纯 Python 泄漏。
- 未发现异常后继续持有大量任务对象的路径。
- 训练结束后 RSS 未立即回到初始值，与原生分配器保留地址空间的现象一致；没有重复阶段增长证据，不能据此认定为泄漏。

## 3. 根因证据

### 3.1 Python 多进程放大

同一份本地基准语料只做 JSONL 读取和文本提取：

| 模式 | 峰值进程树 RSS | 耗时 | 吞吐 |
|---|---:|---:|---:|
| 1 进程 | 21.4 MB | 0.108 s | 50.63 MB/s |
| 8 进程 | 228.7 MB | 1.905 s | 2.87 MB/s |
| `num_workers=0`，实际 20 进程 | 516.7 MB | 2.064 s | 2.65 MB/s |

8 进程峰值约为单进程的 `10.7x`，且读取更慢。`num_workers=0` 在 20 逻辑核机器上达到 `516.7 MB`，与“训练内存爆炸”现象直接吻合。

### 3.2 Tokenizers 原生 BPE 工作集

旧实现单进程训练配置关闭数字拆分以绕过 API 错误后：

- 实际词表停在 3,748
- 峰值进程树 RSS 253.3 MB
- 耗时 5.191 s

用 `RAYON_NUM_THREADS=1` 对照：

- 峰值 230.3 MB
- 耗时 6.931 s

这说明：

- Tokenizers 原生训练确实占用百 MB 级工作集。
- 原生并行在这里只带来约 23 MB 峰值差异，却明显提高吞吐。
- 因此不能无条件关闭并行，也不能把所有增长归因于多进程或泄漏。

### 3.3 Transformers 保存包装

旧流程保存阶段通过 `PreTrainedTokenizerFast` 进入 Transformers 加载路径，保存后仍保留原生 Tokenizer 与包装对象。优化过程中曾重新启用该路径，测得端到端峰值 `413.37 MB`；释放训练实例后仍为 `256.05 MB`，主要工作集来自 Transformers 保存/加载。

改成原生一次保存后：

- 保存阶段增量约 `0.8-1.4 MB`
- 原生重新加载和功能验证阶段增量约 `0.3-3.8 MB`
- 训练进程峰值降至 `58.14 MB`

完整 Transformers + Jinja 验证没有取消，而是拆为独立命令单独测量，避免它在每个训练进程中重复加载。

## 4. 优化架构

新调用关系：

1. `../trainer/train_tokenizer.py`
   - 在导入 Tokenizers 前应用 `--low-memory` 环境变量。
   - 解析和覆盖配置，组织训练。
2. `../tokenizer/config.py`
   - 校验 YAML、文件路径、词表大小、归一化、特殊 Token 顺序和资源参数。
3. `../tokenizer/data_reader.py`
   - JSONL 逐行读取。
   - 顶层 JSON 数组使用 `json.JSONDecoder.raw_decode` 流式读取。
   - 顶层 JSON 对象字段可选 `ijson` 流式读取；小文件需显式允许整对象加载。
   - 只返回 `text_key` 对应文本，不保存原始记录。
4. `../tokenizer/tokenizer.py`
   - 构建 Byte-Level BPE。
   - 在训练前固定特殊 Token 顺序。
   - 使用 `train_from_iterator`。
   - 原生一次保存，显式生成兼容配置。
   - 默认使用 Tokenizers 重新加载验证；可选完整 Transformers 验证。
5. `../tokenizer/resource_monitor.py`
   - 后台采样主进程及全部子进程 RSS。
   - 记录初始化、训练、保存、重载验证的阶段耗时和 RSS 增量。
   - 只保留峰值与阶段聚合，不保留无限时间序列。
6. `../tokenizer/special_tokens.py` / `tokenizer/template.py`
   - 对照配置、实际 added vocabulary、ID 和模板引用。
7. `../tokenizer/validator.py`
   - 快速原生验证：ID、独立编码、解码、可逆性、文件一致性。
   - 完整验证：`PreTrainedTokenizerFast` 重载、单轮/多轮 Jinja 渲染。
8. `../benchmarks`
   - 端到端训练基准。
   - 固定样例分词质量对比。

已经证实的问题对应到确定性优化；可选扩展只有：

- 顶层 JSON 对象字段的 `ijson` 流式读取。
- 训练进程内完整 Transformers 验证开关。
- 更多模板控制 Token 的语义扩展。

## 5. 特殊 Token 与 Chat Template

协议顺序固定为：

| ID | Token | 用途 |
|---:|---|---|
| 0 | `<\|pad\|>` | `pad_token` |
| 1 | `<unk>` | `unk_token` |
| 2 | `<\|endoftext\|>` | 文档结束 |
| 3 | `<\|taichu_start\|>` | `bos_token`，消息开始 |
| 4 | `<\|taichu_end\|>` | `eos_token`，消息结束 |
| 5 | `<\|think\|>` | 思考区段开始 |
| 6 | `<\|/think\|>` | 思考区段结束 |
| 7 | `<\|ghissue\|>` | 项目保留 Token |
| 8 | `<\|final\|>` | 项目保留 Token |

Chat Template 使用 `taichu_start`/`taichu_end` 表示 system、user、assistant 消息边界，并支持：

- system 消息。
- 多轮消息。
- assistant 的 `reasoning_content`。
- `add_generation_prompt`。
- 空消息过滤。

`ghissue` 与 `final` 已在词表中注册，但当前仓库没有定义它们在消息协议中的明确语义，因此模板没有凭空赋予含义。

## 6. 运行方式

安装依赖：

```powershell
& 'D:\Environments\Anaconda\envs\AI\python.exe' -m pip install -r requirements.txt
```

当前工作区中默认配置声明的 `TaiChu/merge_wiki_ch.jsonl` 不存在，因此生产训练必须显式指定真实语料：

```powershell
& 'D:\Environments\Anaconda\envs\AI\python.exe' -m trainer.train_tokenizer `
  --config configs\TaiChu_Tokenizer_config.yaml `
  --data_files D:\path\to\corpus.jsonl `
  --report experiments\tokenizer_memory.json
```

资源受限模式：

```powershell
& 'D:\Environments\Anaconda\envs\AI\python.exe' -m trainer.train_tokenizer `
  --config configs\TaiChu_Tokenizer_config.yaml `
  --data_files D:\path\to\corpus.jsonl `
  --low-memory `
  --report experiments\tokenizer_memory_low.json
```

完整重载与 Jinja 渲染验证：

```powershell
& 'D:\Environments\Anaconda\envs\AI\python.exe' -m trainer.validate_tokenizer `
  --config configs\TaiChu_Tokenizer_config.yaml `
  --tokenizer_dir taichu_tokenizer `
  --vocab_size 32768 `
  --report experiments\tokenizer_full_validation.json
```

自动化测试：

```powershell
& 'D:\Environments\Anaconda\envs\AI\python.exe' -m pytest tests\test_tokenizer.py -q
& 'D:\Environments\Anaconda\envs\AI\python.exe' -m pytest -q
```

基准：

```powershell
& 'D:\Environments\Anaconda\envs\AI\python.exe' benchmarks\benchmark_tokenizer_training.py `
  --data-files D:\path\to\benchmark.jsonl `
  --vocab-size 3722 `
  --min-frequency 3 `
  --output benchmarks\output\tokenizer `
  --report benchmarks\output\report.json

& 'D:\Environments\Anaconda\envs\AI\python.exe' benchmarks\compare_tokenizers.py `
  --before benchmarks\output\before `
  --after benchmarks\output\after `
  --output benchmarks\output\quality.json
```

## 7. 实测性能

以下数据均使用同一份本地确定性基准语料、相同 BPE 路线和 `min_frequency=3`。
旧实现目标词表为 32,000，但语料在 3,748 时耗尽合并；新实现目标为实际可达的
3,722。当前仓库样例产物为 32,768 词表；生产词表仍应使用目标语料重新训练。

| 场景 | 峰值进程树 RSS | 总耗时 | 实际词表 | 说明 |
|---|---:|---:|---:|---|
| 旧实现，1 个 Python 读取进程 | 253.3 MB | 5.191 s | 3,748 | Transformers 保存包装 |
| 旧实现，8 个 Python 读取进程 | 264.1 MB | 7.165 s | 3,748 | 进程更多但更慢 |
| 新实现，标准模式 | 58.14 MB | 0.795 s | 3,722 | Tokenizers 原生保存/重载 |
| 新实现，低内存模式 | 38.72 MB | 2.157 s | 3,722 | `RAYON_NUM_THREADS=1` |
| 独立完整 Transformers 验证 | 235.93 MB | 6.395 s | 3,722 | 不计入训练峰值 |

训练阶段内部数据：

| 模式 | 训练阶段耗时 | 训练阶段 RSS 增量 | 训练吞吐 |
|---|---:|---:|---:|
| 标准模式 | 0.681 s | 21.88 MB | 8.42 M chars/s |
| 低内存模式 | 2.058 s | 4.27 MB | 2.79 M chars/s |

质量对比：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 固定样例总 Token 数 | 216 | 220 |
| 可逆样例 | 5/7 | 7/7 |
| 中文样例 Token 数 | 22 / 22 | 22 / 21 |
| 中英混合样例 Token 数 | 51 | 51 |
| 代码样例 Token 数 | 23 | 23 |
| Emoji 样例 Token 数 | 38 | 39 |
| 空行、换行、缩进样例 | 26，不可逆 | 29，可逆 |

质量结论：

- 中文没有出现整体过度拆分。
- 英文和混合文本 Token 数基本稳定。
- 总 Token 微增来自保留原始空白，这是正确性修复。
- 代码和空白样例从不可逆变为可逆。

## 8. 验证结果

已执行：

- Tokenizer 专项测试：`14 passed`
- 全项目测试：`154 passed, 1 skipped, 120 subtests passed`
- Python 语法编译：通过
- 完整 Transformers 重载：通过
- 单轮和多轮 Chat Template 渲染：通过
- 特殊 Token ID 0–8：通过
- 保存后重新加载：通过
- 中文、英文、混合、数字、标点、代码、空白、Emoji、空字符串、长文本：通过
- JSONL、顶层 JSON 数组、显式小 JSON 对象：通过

全项目测试只有既有的 PyTorch JIT 弃用警告，没有失败。

## 9. 风险与待验证项

### 已修复：Tokenizer 文件协议不一致

当前 `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json` 和
Chat Template 已统一到 9 个特殊 Token。原生 Tokenizers 重载、完整 Transformers
重载、Chat Template 渲染和随机 Unicode 往返均通过。

### 已修复：模型 Embedding 兼容性

当前内置 Tokenizer 和所有样例模型配置都使用 `vocab_size=32768`。
`TokenizerProcessor` 和训练/评估入口也会检查 Tokenizer 词表不超过模型词表。

### P1：生产 32k 词表仍需目标语料实测

当前 32,768 词表产物已通过正确性验证，但它的压缩率和领域覆盖不代表最终生产质量。
正式训练前应使用目标中英文/代码语料重新训练，并记录实际峰值、耗时、bytes/token、
特殊 Token 和代表性质量样例。

### P1：严格词表大小校验

训练不再静默保存小于目标词表的 Tokenizer。如果语料多样性不足或 `min_frequency` 过高，训练会明确失败。这避免 Token ID 与模型 Embedding 不匹配，但正式训练前需要确认参数。

### P1：Chat Template 语义

`ghissue`、`final` 当前只有预留 Token，没有项目明确语义，未加入模板。若它们应参与工具调用、最终答案或控制权交接，必须先给出协议定义，再扩展模板和测试。

### P2：JSON 对象字段

顶层 JSON 数组可以无额外依赖流式读取。顶层 JSON 对象字段建议安装 `ijson`；否则仅允许显式开启小文件完整加载，避免为了一个字段构建大型 Python 对象树。

### P2：完整验证内存

完整 Transformers 验证独立运行时峰值约 236 MB，不应与训练进程内并行执行。默认训练使用 Tokenizers 原生重载验证；提交或发布 Tokenizer 前再运行全量验证。
