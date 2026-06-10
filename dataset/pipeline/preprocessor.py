#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 语料预处理流水线 (Production-Ready)

功能模块：
1. 清洗 (Clean)：敏感词/低质内容剔除（语言识别已移除）
2. 过滤 (Filter)：长度、符号比例、乱码检测、PII 消除、困惑度评估（HuggingFace）
3. 标准化 (Normalize)：全半角转换、繁简转换、空白符压缩
4. 去重 (Deduplicate)：精确去重（哈希）与模糊去重（SimHash）

参考项目：
- Data-Juicer: https://github.com/modelscope/data-juicer
- CCNet: https://github.com/facebookresearch/cc_net
- Text-Dedup: https://github.com/ChenghaoMou/text-dedup
- OpenCC: https://github.com/BYVoid/OpenCC

依赖：
transformers>=4.30.0, torch>=2.11.0, langdetect>=1.0.9, pycld2>=0.41 (可选),
opencc-python-reimplemented>=0.1.7, simhash>=2.1.2, jieba>=0.42.1, tqdm>=4.64.0

使用示例：
---------
1. 最简模式（仅基础清洗+标准化+去重，禁用PII和困惑度）
   python -m dataset.pipeline.preprocess --input input.txt --output output.txt --no-pii --no-perplexity

2. 仅启用PII移除，不启用困惑度
   python -m dataset.pipeline.preprocess --input input.txt --output output.txt --no-perplexity

3. 仅启用困惑度评估（使用distilgpt2，阈值80），禁用PII
   python -m dataset.pipeline.preprocess --input input.txt --output output.txt --no-pii \\
       --hf-model distilgpt2 --max-perplexity 80 --device cpu

4. 同时启用PII移除和困惑度评估
   python preprocess.py --input input.txt --output output.txt \\
       --hf-model distilgpt2 --max-perplexity 100 --device cpu

5. 高级自定义（指定长度、仅精确去重等）
   python preprocess.py --input input.txt --output output.txt \\
       --min-chars 100 --max-chars 50000 \\
       --no-pii --hf-model gpt2 --max-perplexity 120 --device cpu \\
       --to-simplified --exact-dedup-only

6. GPU加速困惑度（需CUDA环境）
   python preprocess.py --input input.txt --output output.txt \\
       --hf-model distilgpt2 --max-perplexity 90 --device cuda

参数说明：
  --no-pii              禁用PII（邮箱/身份证/手机号等）移除
  --no-perplexity       完全禁用困惑度计算（无需提供--hf-model）
  --hf-model MODEL      启用困惑度时使用的HuggingFace模型名称
  --max-perplexity N    困惑度阈值，超过则丢弃文档
  --device {cpu,cuda}   计算设备，默认cpu
  --to-simplified       繁体转简体（默认开启，加--no-simplified关闭）
  --exact-dedup-only    仅精确去重，跳过SimHash模糊去重
"""

import argparse
import hashlib
import logging
import re
import string
import unicodedata
import json
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple
from multiprocessing import Pool

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from simhash import Simhash, SimhashIndex

from utils.logger import LogManager

# 用于多进程 worker 接收参数
_GLOBAL_ARGS = None

# 尝试导入 OpenCC（繁简转换），若未安装则降级处理
try:
    from opencc import OpenCC

    _OPENCC_AVAILABLE = True
except ImportError:
    _OPENCC_AVAILABLE = False
    logging.warning("OpenCC 未安装，繁简转换功能将降级为空操作。请执行: pip install opencc-python-reimplemented")

# ---------- 数据结构 ----------
@dataclass
class Document:
    """标准化文档结构，用于在流水线各阶段传递数据。"""

    text: str
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:50].replace('\n', '\\n')
        return f"Document(text={preview}..., meta={self.meta})"


# ---------- 抽象基类 ----------
class BaseProcessor(ABC):
    """所有处理器的抽象基类，统一接口。"""

    @abstractmethod
    def process(self, docs: Iterable[Document]) -> Iterable[Document]:
        """对文档流进行处理并返回结果流。"""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ---------- 辅助函数 ----------
def count_lines_fast(file_path: str) -> int:
    """快速统计文件行数（适用于 UTF-8 编码）。"""
    count = 0
    with open(file_path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def read_documents(input_path: str, input_format: str = 'auto') -> List[Document]:
    """
    读取输入文件，支持 txt（每行纯文本）和 jsonl（每行 JSON，需包含 'text' 字段）。

    Args:
        input_path: 文件路径
        input_format: 'txt', 'jsonl', 或 'auto'（根据扩展名自动判断）

    Returns:
        Document 列表
    """
    path = Path(input_path)
    if input_format == 'auto':
        if path.suffix.lower() == '.jsonl':
            input_format = 'jsonl'
        else:
            input_format = 'txt'

    # 快速统计总行数（用于进度条）
    total_lines = count_lines_fast(input_path)

    docs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_format == 'jsonl':
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Reading jsonl", unit="line"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if not text:
                        continue
                    meta = {k: v for k, v in data.items() if k != 'text'}
                    docs.append(Document(text=text, meta=meta))
                except json.JSONDecodeError:
                    logging.warning(f"第 {line_num} 行 JSON 解析失败，已跳过")
        else:  # txt
            for line in tqdm(f, total=total_lines, desc="Reading txt", unit="line"):
                line = line.strip()
                if line:
                    docs.append(Document(text=line, meta={}))
    return docs


def write_documents(docs: Iterable[Document], output_path: str, output_format: str = 'auto') -> int:
    """
    将文档写入文件，支持 txt（每行纯文本）和 jsonl（每行 JSON，包含 text 和 meta）。

    Args:
        docs: 文档迭代器
        output_path: 输出文件路径
        output_format: 'txt', 'jsonl', 或 'auto'（根据扩展名自动判断）

    Returns:
        写入的文档数量
    """
    path = Path(output_path)
    if output_format == 'auto':
        if path.suffix.lower() == '.jsonl':
            output_format = 'jsonl'
        else:
            output_format = 'txt'

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_format == 'jsonl':
            for doc in tqdm(docs, desc="Writing jsonl", unit="doc"):
                out_obj = {'text': doc.text}
                if doc.meta:
                    out_obj.update(doc.meta)
                f.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
                count += 1
        else:  # txt
            for doc in tqdm(docs, desc="Writing txt", unit="doc"):
                f.write(doc.text + '\n')
                count += 1
    return count


def split_docs_by_size(docs: List[Document], max_size_mb: float) -> List[List[Document]]:
    """
    将文档列表按估算的数据量切分成多个块，每个块的总大小（序列化后近似值）不超过 max_size_mb。

    估算规则：len(doc.text)（字符数）≈ 字节数（UTF-8 编码），
    加上 JSON 字段开销（约 100 字节/文档），最终乘以 1.2 安全系数。

    Args:
        docs: 文档列表
        max_size_mb: 单块最大大小（MB）

    Returns:
        块列表，每个块是 Document 子列表
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    # 每个文档的近似开销：文本字节数 + JSON 结构开销（100 字节）
    # 为了安全，再乘以 1.1 系数，避免序列化后超过实际限制
    overhead_per_doc = 100
    safe_factor = 1.1

    chunks = []
    current_chunk = []
    current_size = 0

    for doc in docs:
        # 估算该文档的大小：UTF-8 字节数 + 开销
        doc_size = len(doc.text.encode('utf-8')) + overhead_per_doc
        doc_size = int(doc_size * safe_factor)

        # 如果单个文档超过上限，单独成块
        if doc_size >= max_size_bytes:
            # 先保存当前块（非空）
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            chunks.append([doc])
            continue

        # 如果加入当前文档会超过上限，则结束当前块并新建
        if current_size + doc_size > max_size_bytes:
            chunks.append(current_chunk)
            current_chunk = [doc]
            current_size = doc_size
        else:
            current_chunk.append(doc)
            current_size += doc_size

    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def _is_cjk(uchar: str) -> bool:
    """判断一个 Unicode 字符是否属于 CJK（中日韩统一表意文字）。"""
    cp = ord(uchar)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def _is_fullwidth(uchar: str) -> bool:
    """判断字符是否为全角形式（FF00-FFEF）。"""
    return 0xFF01 <= ord(uchar) <= 0xFF60


def _normalize_fullwidth_halfwidth(text: str) -> str:
    """全角字符转半角字符，仅处理常见标点和字母数字。"""
    result = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            result.append(' ')
        elif 0xFF01 <= code <= 0xFF5E:  # 全角字母数字符号区间
            result.append(chr(code - 0xFEE0))
        else:
            result.append(ch)
    return ''.join(result)


def _normalize_whitespace(text: str) -> str:
    """将所有空白字符（包括全角空格、制表符等）压缩为单个半角空格，并去除首尾空格。"""
    text = text.replace('\u3000', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_junk_character_ratio(text: str, threshold: float = 0.3) -> bool:
    """检查文本中不可打印字符或乱码的比例是否超过阈值。"""
    if not text:
        return True
    junk_count = sum(1 for ch in text if unicodedata.category(ch) in ('So', 'Cn', 'Co'))
    return (junk_count / len(text)) > threshold


# ---------- 1. 清洗模块 ----------
class CleanProcessor(BaseProcessor):
    """清洗阶段：敏感词过滤、低质内容剔除（语言识别已移除）。"""

    def __init__(
        self,
        blocked_words: Optional[Set[str]] = None,
    ) -> None:
        """初始化清洗器。

        Args:
            blocked_words: 敏感词黑名单集合。
        """
        self.blocked_words = blocked_words or set()
        if self.blocked_words:
            # 预编译敏感词正则，忽略大小写
            escaped = (re.escape(w) for w in self.blocked_words)
            self._blocked_pattern = re.compile('|'.join(escaped), re.IGNORECASE)
        else:
            self._blocked_pattern = None

    def process(self, docs: Iterable[Document]) -> Iterable[Document]:
        """依次处理文档，进行敏感词过滤。"""
        for doc in docs:
            # 敏感词过滤：如果命中任一敏感词，整篇丢弃
            if self._blocked_pattern and self._blocked_pattern.search(doc.text):
                continue

            yield doc


# ---------- 2. 过滤模块 ----------
class FilterProcessor(BaseProcessor):
    """多维度质量过滤：长度、符号比例、乱码、PII 消除、困惑度（HuggingFace 模型）。"""

    def __init__(
        self,
        min_chars: int = 50,
        max_chars: int = 100000,
        min_symbol_ratio: float = 0.0,
        max_symbol_ratio: float = 0.5,
        use_pii_removal: bool = True,
        use_perplexity: bool = False,
        hf_model_name: Optional[str] = None,
        max_perplexity: Optional[float] = None,
        device: str = 'cpu',
        cache_dir: Optional[str] = None,
    ) -> None:
        """初始化过滤器。

        Args:
            min_chars: 最小字符数（去除首尾空格后）。
            max_chars: 最大字符数。
            min_symbol_ratio: 标点符号占文本的最小比例。
            max_symbol_ratio: 标点符号占文本的最大比例。
            use_pii_removal: 是否移除邮箱、身份证号等隐私信息。
            use_perplexity: 是否计算困惑度。
            hf_model_name: HuggingFace 语言模型名称，如 'distilgpt2'，用于困惑度计算。
            max_perplexity: 最大允许困惑度，超过此值的文档被丢弃。
            device: 计算设备，'cpu' 或 'cuda'，Windows 推荐 'cpu'。
        """
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_symbol_ratio = min_symbol_ratio
        self.max_symbol_ratio = max_symbol_ratio
        self.use_pii_removal = use_pii_removal
        self.use_perplexity = use_perplexity
        self.max_perplexity = max_perplexity
        self.device = device

        # PII 正则（仅在需要时保留，避免无用编译）
        self._pii_patterns = None
        if self.use_pii_removal:
            self._pii_patterns = [
                re.compile(r'\b[\w.-]+@[\w.-]+\.\w{2,}\b'),
                re.compile(r'\b\d{15,18}\b'),
                re.compile(r'\b1[3-9]\d{9}\b'),
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            ]

        # 初始化 HuggingFace 模型（若提供）
        self._hf_tokenizer = None
        self._hf_model = None
        if self.use_perplexity and hf_model_name:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            # 若分词器无 pad_token，设置为 eos_token
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype=torch.float32,  # CPU 兼容
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            ).to(device)
            self._hf_model.eval()

    def _symbol_ratio(self, text: str) -> float:
        """计算标点符号占总字符数的比例。"""
        if not text:
            return 0.0
        count = sum(1 for ch in text if ch in string.punctuation or _is_fullwidth(ch))
        return count / len(text)

    def _remove_pii(self, text: str) -> str:
        """用 [PII] 替换常见隐私信息。"""
        for pattern in self._pii_patterns:
            text = pattern.sub('[PII]', text)
        return text

    def _perplexity(self, text: str) -> Optional[float]:
        """使用 HuggingFace 因果语言模型计算困惑度。"""
        if not self._hf_model:
            return None
        # 截断过长文本，避免内存爆炸
        encodings = self._hf_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._hf_model(encodings.input_ids, labels=encodings.input_ids)
            loss = outputs.loss
            if loss is not None:
                return torch.exp(loss).item()
        return None

    def process(self, docs: Iterable[Document]) -> Iterable[Document]:
        """依次过滤文档。"""
        for doc in docs:
            text = doc.text

            # 长度过滤
            char_len = len(text)
            if char_len < self.min_chars or char_len > self.max_chars:
                continue

            # 符号比例过滤
            sym_ratio = self._symbol_ratio(text)
            if sym_ratio < self.min_symbol_ratio or sym_ratio > self.max_symbol_ratio:
                continue

            # 乱码检测
            if _is_junk_character_ratio(text):
                continue

            # PII 消除
            if self.use_pii_removal:
                text = self._remove_pii(text)

            # 困惑度过滤（若启用）
            if self.use_perplexity and self._hf_model and self.max_perplexity is not None:
                ppl = self._perplexity(text)
                if ppl is not None and ppl > self.max_perplexity:
                    continue
                doc.meta['perplexity'] = ppl

            doc.text = text
            yield doc


# ---------- 3. 标准化模块 ----------
class NormalizeProcessor(BaseProcessor):
    """文本标准化：繁简转换、全半角转换、空白符压缩、Unicode 规范化。"""

    def __init__(self, to_simplified: bool = True, unicode_normalize: str = 'NFKC') -> None:
        """初始化标准化处理器。

        Args:
            to_simplified: True 为繁体转简体，False 为简体转繁体。
            unicode_normalize: Unicode 规范化形式，推荐 'NFKC'。
        """
        self.to_simplified = to_simplified
        self.unicode_normalize = unicode_normalize
        if _OPENCC_AVAILABLE:
            self._opencc = OpenCC('t2s' if to_simplified else 's2t')
        else:
            self._opencc = None

    def process(self, docs: Iterable[Document]) -> Iterable[Document]:
        """依次对文档进行标准化。"""
        for doc in docs:
            text = doc.text

            # Unicode 规范化（合并全角/半角、合字等）
            text = unicodedata.normalize(self.unicode_normalize, text)

            # 繁简转换
            if self._opencc:
                text = self._opencc.convert(text)

            # 残留全角标点转半角
            text = _normalize_fullwidth_halfwidth(text)

            # 空白符压缩
            text = _normalize_whitespace(text)

            doc.text = text
            yield doc


# ---------- 4. 去重模块 ----------
class DeduplicateProcessor(BaseProcessor):
    """文档级去重：精确去重（MD5） + SimHash 模糊去重。

    注意：此处理器会消耗上游迭代器，将所有文档加载到内存构建索引，
    适合中等规模数据集（数百万级）。超大规模数据建议结合外部存储。
    """

    def __init__(
        self,
        simhash_threshold: int = 6,
        simhash_blocks: int = 64,
        only_exact: bool = False,
    ) -> None:
        """初始化去重器。

        Args:
            simhash_threshold: 汉明距离阈值，≤此值视为重复。
            simhash_blocks: SimHash 索引分块数，越大召回率越高，但内存开销越大。
            only_exact: 若为 True，仅执行精确去重，跳过 SimHash。
        """
        self.simhash_threshold = simhash_threshold
        self.simhash_blocks = simhash_blocks
        self.only_exact = only_exact
        self._exact_set: Set[str] = set()
        self._sim_index: Optional[SimhashIndex] = None

    def _compute_simhash(self, text: str) -> Simhash:
        """计算文本的 SimHash 指纹。

        对中文进行字符级分词，对英文按空格分词。
        """
        if any(_is_cjk(ch) for ch in text[:200]):
            tokens = list(text)  # 字符级
        else:
            tokens = text.lower().split()
        return Simhash(tokens)

    def process(self, docs: Iterable[Document]) -> Iterable[Document]:
        """执行去重，先精确后模糊。"""
        # 先将所有文档读入列表以获取总长度（去重需要两遍扫描）
        docs_list = list(docs)

        # 第一阶段：缓存所有文档，同时进行精确去重
        buffer: List[Document] = []
        for doc in tqdm(docs_list, desc="Exact dedup", unit="doc"):
            text = doc.text
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in self._exact_set:
                continue
            self._exact_set.add(text_hash)
            buffer.append(doc)

        # 第二阶段：模糊去重
        if not self.only_exact and buffer:
            # 构建 SimHash 索引
            sim_objs = []
            for i, doc in enumerate(tqdm(buffer, desc="Computing Simhash", unit="doc")):
                sim_objs.append((self._compute_simhash(doc.text), i))
            self._sim_index = SimhashIndex(
                sim_objs, k=self.simhash_threshold, f=self.simhash_blocks
            )
            kept_indices: Set[int] = set()
            for i, doc in enumerate(tqdm(buffer, desc="Fuzzy dedup", unit="doc")):
                sim = sim_objs[i][0]
                near_dups = self._sim_index.get_near_dups(sim)
                # 保留簇中最早出现的文档（索引最小）
                if not near_dups or min(near_dups) == i:
                    kept_indices.add(i)
            for i in kept_indices:
                yield buffer[i]
        else:
            yield from buffer


# ---------- 5. 流水线编排 ----------
class PreprocessingPipeline:
    """将多个处理器串联为一条流水线，支持链式调用。"""

    def __init__(self, processors: Optional[List[BaseProcessor]] = None) -> None:
        self.processors = processors or []

    def add_processor(self, processor: BaseProcessor) -> 'PreprocessingPipeline':
        """添加处理器，返回 self 以支持链式调用。"""
        self.processors.append(processor)
        return self

    def run(self, docs: Iterable[Document]) -> Iterable[Document]:
        """顺序执行所有处理器并返回最终结果流。"""
        stream = docs
        if not isinstance(stream, list):
            stream = list(stream)

        for proc in self.processors:
            iterator = tqdm(stream, desc=str(proc), unit='doc', leave=False)
            stream = list(proc.process(iterator))
            logging.info(f"{proc} 完成，剩余文档数: {len(stream)}")

        return stream


def process_chunk(args, chunk: List[Document]) -> List[Document]:
    """
    在单个子进程中处理一个文档块，顺序执行：
        清洗（CleanProcessor） -> 过滤（FilterProcessor） -> 标准化（NormalizeProcessor）

    注意：每个子进程会独立加载 OpenCC、HuggingFace 模型等资源，
         因此在主进程初始化时需控制 chunk 大小和进程数，避免内存爆炸。

    为规避多进程下进度条闪烁问题，所有子进程内的 tqdm 进度条均设置为禁用。

    Args:
        args: 命令行参数对象（包含所有配置开关，如 --no-pii 等）
        chunk: 文档列表（原始文本，尚未经过任何处理）

    Returns:
        处理后的文档列表（已完成清洗、过滤、标准化，尚未去重）
    """
    proc = multiprocessing.current_process()
    proc_id = proc._identity[0] if proc._identity else 0

    # 多进程子进程：禁用进度条；主进程（单进程模式）开启进度条
    disable_progress = (proc_id >= 1)

    # 1. 创建清洗处理器（语言识别已移除）
    clean_proc = CleanProcessor(
        blocked_words=set(),  # 可在此扩展敏感词列表
    )

    # 2. 创建过滤处理器
    # 注意：若启用了困惑度，每个子进程都会独立加载 HF 模型，会增加内存和启动时间
    filter_proc = FilterProcessor(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_symbol_ratio=args.min_symbol_ratio,
        max_symbol_ratio=args.max_symbol_ratio,
        use_pii_removal=not args.no_pii,
        use_perplexity=(not args.no_perplexity and args.hf_model is not None),
        hf_model_name=args.hf_model,
        max_perplexity=args.max_perplexity,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # 3. 创建标准化处理器
    norm_proc = NormalizeProcessor(to_simplified=args.to_simplified)

    # 4. 依次处理
    stream = chunk
    # 清洗阶段进度条
    stream = list(tqdm(clean_proc.process(stream), desc="[Clean]", total=len(stream), disable=disable_progress))
    # 过滤阶段进度条
    stream = list(tqdm(filter_proc.process(stream), desc="[Filter]", total=len(stream), disable=disable_progress))
    # 标准化阶段进度条
    stream = list(tqdm(norm_proc.process(stream), desc="[Normalize]", total=len(stream), disable=disable_progress))

    return stream


def init_worker(shared_args):
    """子进程初始化函数，用于接收父进程传递的参数。"""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = shared_args


def process_chunk_worker(chunk: List[Document]) -> List[Document]:
    """供 Pool.imap_unordered 调用的包装函数。"""
    global _GLOBAL_ARGS
    return process_chunk(_GLOBAL_ARGS, chunk)


# ---------- 命令行参数解析 ----------
def parse_args():
    """解析命令行参数（已移除 --target-langs 语言识别相关参数）。"""
    parser = argparse.ArgumentParser(description='LLM 语料预处理流水线')
    parser.add_argument('--input', required=False, help='输入文件路径，每行一个文档')
    parser.add_argument('--output', required=False, help='输出文件路径')
    parser.add_argument('--input-format', choices=['auto', 'txt', 'jsonl'], default='auto',
                        help='输入格式，auto 根据扩展名自动判断')
    parser.add_argument('--output-format', choices=['auto', 'txt', 'jsonl'], default='auto',
                        help='输出格式，auto 根据扩展名自动判断')
    parser.add_argument('--min-chars', type=int, default=50, help='最小字符数')
    parser.add_argument('--max-chars', type=int, default=100000, help='最大字符数')
    parser.add_argument('--min-symbol-ratio', type=float, default=0.01, help='最小标点符号比例')
    parser.add_argument('--max-symbol-ratio', type=float, default=0.4, help='最大标点符号比例')
    parser.add_argument('--no-pii', action='store_true', help='禁用 PII 消除')
    parser.add_argument('--no-perplexity', action='store_true', help='禁用困惑度计算与过滤')
    parser.add_argument('--hf-model', default=None, help='HuggingFace 模型名，如 distilgpt2（需启用困惑度时提供）')
    parser.add_argument('--max-perplexity', type=float, default=None, help='最大困惑度阈值（仅困惑度启用时有效）')
    parser.add_argument('--device', default='cpu', help='计算设备：cpu 或 cuda')
    parser.add_argument('--to-simplified', action='store_true', default=True, help='繁体转简体')
    parser.add_argument('--no-simplified', dest='to_simplified', action='store_false', help='保持原文繁简')
    parser.add_argument('--exact-dedup-only', action='store_true', help='仅精确去重')
    parser.add_argument('--num-workers', type=int, default=8, help='并行处理的进程数（默认1，单进程；建议不超过8）')
    parser.add_argument('--chunk-size-mb', type=float, default=100, help='每个 worker 处理的数据块大小上限（MB），默认 50MB。单文档超过上限时独自成块。')
    parser.add_argument('--log-dir', default='./experiments/logs', help='日志目录')
    parser.add_argument('--no-tensorboard', action='store_true', help='禁用 TensorBoard')
    parser.add_argument('--cache-dir', default='./experiments/cache', help='HuggingFace 缓存目录')
    return parser.parse_args()


if __name__ == "__main__":
    # ---------- 解析参数 ----------
    args = parse_args()
    # 将 args 设置为全局变量，供子进程包装函数使用
    _GLOBAL_ARGS = args

    # ---------- 初始化日志 ----------
    log_manager = LogManager(
        log_dir=args.log_dir,
        tensorboard=False,
        log_file="dat_preprocess.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )

    # ---------- 读取文档 ----------
    # 记录输入输出文件信息
    log_manager.info(f"开始处理文档: {args.input}")
    log_manager.info(f"输出路径: {args.output}")
    docs = read_documents(args.input, args.input_format)
    log_manager.info(f'原始文档数量: {len(docs)}')

    # ---------- 多进程并行处理（清洗+过滤+标准化）----------
    # 决定是否使用多进程（可通过命令行参数控制，默认单进程）
    num_workers = getattr(args, 'num_workers', 1)
    if num_workers <= 1:
        # 单进程处理
        processed_docs = process_chunk(args, docs)
    else:
        # 多进程处理
        num_workers = min(num_workers, multiprocessing.cpu_count(), 8)
        log_manager.info(f'使用 {num_workers} 个进程并行处理')

        # 切分成更小的块，避免 pickling 过大
        # 使用按大小切分
        chunks = split_docs_by_size(docs, args.chunk_size_mb)
        log_manager.info(f'使用 {num_workers} 个进程，共 {len(chunks)} 个块')

        with Pool(num_workers, initializer=init_worker, initargs=(args,)) as pool:
            try:
                # 使用 map 或 imap_unordered，这里使用 imap_unordered 并显示进度
                results = []
                for result_chunk in tqdm(pool.imap_unordered(process_chunk_worker, chunks),
                                         total=len(chunks), desc="Parallel processing", unit="chunk"):
                    results.extend(result_chunk)
                processed_docs = results
            finally:
                pool.terminate()  # 强制终止所有子进程
                pool.join()  # 等待子进程退出
                pool.close()

    log_manager.info(f'并行处理后剩余文档数: {len(processed_docs)}')

    # ---------- 单线程去重（精确去重 + SimHash）----------
    dedup_proc = DeduplicateProcessor(only_exact=args.exact_dedup_only)
    final_docs = list(dedup_proc.process(processed_docs))
    log_manager.info(f'去重后最终文档数: {len(final_docs)}')

    # ---------- 写入输出文件 ----------
    count = write_documents(final_docs, args.output, args.output_format)
    log_manager.info(f'处理完成，保留文档数量: {count}，输出至 {args.output}')

    # 关闭日志管理器
    log_manager.close()