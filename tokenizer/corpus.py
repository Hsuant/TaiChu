# -*- coding: utf-8 -*-
"""数据集迭代器模块（内存优化版）。

该模块提供 CorpusIterator 类，用于从 JSONL 格式文件中流式读取文本数据，
支持可选的并行处理（多进程）和中文分词（jieba / pkuseg）。
主要用于 BPE 分词器训练前的语料预处理。

优化点：
    - 避免一次性读入整个文件，采用逐块（chunk）读取，显著降低主进程内存峰值。
    - 多进程模式下仍保持流式处理，通过 imap 逐块提交任务，内存开销稳定。
"""

import json
import re
from multiprocessing import Pool, cpu_count
from typing import Iterator, List, Optional

# 尝试导入可选的分词库，并标记是否可用
try:
    import pkuseg
    PKUSEG_AVAILABLE = True
except ImportError:
    PKUSEG_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


# ==================== 多进程全局变量 ====================
# 这些全局变量将在每个子进程中独立存在，通过 _init_worker 进行初始化。

_worker_seg = None                # 分词器实例（jieba 模块或 pkuseg 对象）
_worker_seg_model = None          # 分词模型名称，如 "jieba" 或 "mixed"
_worker_text_key = None           # JSON 对象中文本字段的键名
_worker_split_chinese = None      # 是否启用中文分词的布尔标志
_worker_max_text_length = None    # 单条文本最大长度（字符数）
_worker_skip_empty = None         # 是否跳过空文本
_worker_chinese_pattern = None    # 预编译的中文字符正则表达式


def _init_worker(seg_model: str, text_key: str, split_chinese: bool,
                 max_text_length: Optional[int], skip_empty: bool) -> None:
    """在每个子进程启动时调用，初始化全局配置和分词器。

    该函数作为 multiprocessing.Pool 的 initializer 参数传入，
    会在每个工作进程创建后、开始处理任务之前执行一次。

    Args:
        seg_model: 分词器类型，例如 "jieba" 或 "mixed"/"news"/"default"。
        text_key: JSON 中文本字段的键名。
        split_chinese: 是否启用中文分词。
        max_text_length: 单条文本的最大长度限制（字符数）。
        skip_empty: 是否跳过空文本或仅含空白字符的文本。
    """
    global _worker_seg, _worker_seg_model, _worker_text_key, _worker_split_chinese
    global _worker_max_text_length, _worker_skip_empty, _worker_chinese_pattern

    _worker_seg_model = seg_model
    _worker_text_key = text_key
    _worker_split_chinese = split_chinese
    _worker_max_text_length = max_text_length
    _worker_skip_empty = skip_empty

    # 预编译中文字符匹配正则
    _worker_chinese_pattern = re.compile(
        r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f'
        r'\u2b740-\u2b81f\u2b820-\u2ceaf\u2ceb0-\u2ebef\u30000-\u3134f]'
    )

    # 根据 seg_model 初始化对应的分词器实例
    if seg_model == "jieba":
        if JIEBA_AVAILABLE:
            _worker_seg = jieba
        else:
            raise ImportError("jieba not installed")
    else:
        if PKUSEG_AVAILABLE:
            model_name = seg_model if seg_model in ("mixed", "news", "default") else "mixed"
            _worker_seg = pkuseg.pkuseg(model_name=model_name)
        else:
            raise ImportError("pkuseg not installed")


def _process_line_worker(line: str) -> Optional[str]:
    """供多进程调用的处理函数，使用全局配置处理单行 JSON 数据。

    Args:
        line: 原始字符串行，应为 JSON 格式。

    Returns:
        处理后的文本字符串（若经过分词，单词间以空格分隔），
        如果数据无效则返回 None。
    """
    global _worker_seg, _worker_text_key, _worker_split_chinese, _worker_max_text_length
    global _worker_skip_empty, _worker_chinese_pattern

    line = line.strip()
    if not line:
        return None

    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    text = obj.get(_worker_text_key)
    if not isinstance(text, str):
        return None

    if _worker_skip_empty and not text.strip():
        return None

    if _worker_max_text_length is not None and len(text) > _worker_max_text_length:
        text = text[:_worker_max_text_length]

    if _worker_split_chinese and _worker_chinese_pattern.search(text):
        words = _worker_seg.cut(text)
        text = ' '.join(words)

    return text


class CorpusIterator:
    """可迭代的语料库，逐条生成训练文本。

    该类采用**流式分块**策略：
        - 单进程模式：逐行读取并 yield，内存占用极小。
        - 多进程模式：按行块（chunk）读取，每块作为一个任务提交给进程池，
          主进程仅保留当前块，内存峰值 = 块大小 × 单行平均长度。

    Attributes:
        files: JSONL 文件路径列表。
        epoch: 遍历整个数据集的次数（轮数）。
        text_key: JSON 对象中文本字段的名称。
        split_chinese: 是否对中文文本进行分词。
        seg_model: 分词模型名称。
        max_text_length: 单条文本允许的最大字符数。
        num_workers: 并行处理的进程数，0 表示自动使用 CPU 核心数。
        skip_empty: 是否跳过空文本或仅含空白字符的文本。
        chunk_size: 多进程模式下每个任务包含的行数（默认 5000）。
    """

    def __init__(
        self,
        files: List[str],
        epoch: int = 1,
        text_key: str = "text",
        split_chinese: bool = False,
        seg_model: str = "mixed",
        max_text_length: Optional[int] = None,
        num_workers: int = 0,
        skip_empty: bool = True,
        chunk_size: int = 5000,
    ) -> None:
        """初始化语料迭代器。

        Args:
            files: JSONL 文件路径列表。
            epoch: 数据集遍历轮数，默认为 1。
            text_key: JSON 对象中文本字段的键名。
            split_chinese: 是否启用中文分词。
            seg_model: 分词模型名称（当 split_chinese=True 时有效）。
            max_text_length: 最大文本长度（字符数），None 表示不限制。
            num_workers: 并行进程数，0 表示自动使用 CPU 核心数。
            skip_empty: 是否跳过空文本或仅含空白字符的文本。
            chunk_size: 多进程模式下每个任务包含的行数（块大小）。
        """
        self.files = files
        self.epoch = epoch
        self.text_key = text_key
        self.split_chinese = split_chinese
        self.seg_model = seg_model
        self.max_text_length = max_text_length
        self.num_workers = num_workers if num_workers > 0 else cpu_count()
        self.skip_empty = skip_empty
        self.chunk_size = chunk_size

    def _file_generator_single(self, file_path: str) -> Iterator[str]:
        """单进程模式：顺序读取单个文件并生成处理后的文本。

        Args:
            file_path: JSONL 文件路径。

        Yields:
            经过清洗和分词后的文本字符串。
        """
        # 初始化分词器（与多进程 worker 中的逻辑保持一致）
        if self.seg_model == "jieba":
            seg = jieba
        else:
            model_name = self.seg_model if self.seg_model in ("mixed", "news", "default") else "mixed"
            seg = pkuseg.pkuseg(model_name=model_name)

        chinese_pattern = re.compile(
            r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f'
            r'\u2b740-\u2b81f\u2b820-\u2ceaf\u2ceb0-\u2ebef\u30000-\u3134f]'
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = obj.get(self.text_key)
                if not isinstance(text, str):
                    continue
                if self.skip_empty and not text.strip():
                    continue
                if self.max_text_length and len(text) > self.max_text_length:
                    text = text[:self.max_text_length]
                if self.split_chinese and chinese_pattern.search(text):
                    words = seg.cut(text)
                    text = ' '.join(words)
                yield text

    def _chunk_generator(self, file_path: str) -> Iterator[List[str]]:
        """生成文件的行块（chunk），每个块是一个字符串列表。

        逐行读取文件，累积到 self.chunk_size 行时 yield 该块，
        最后不足一个块的行也作为一个块返回。

        Args:
            file_path: JSONL 文件路径。

        Yields:
            包含多行字符串的列表（每个元素是原始 JSON 行）。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                chunk.append(line)
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

    def __iter__(self) -> Iterator[str]:
        """返回迭代器，支持多轮 epoch 和多进程并行处理。

        多进程模式下，采用流式分块：
            - 主进程从文件中逐块读取行列表（内存中仅存当前块）。
            - 每个块通过 pool.imap 提交给工作进程并行处理。
            - imap 返回的结果流式产出，不会一次性加载所有结果。

        Yields:
            经过清洗和分词后的文本字符串。
        """
        for _ in range(self.epoch):
            for file_path in self.files:
                if self.num_workers > 1:
                    # 多进程流式分块模式
                    with Pool(
                        self.num_workers,
                        initializer=_init_worker,
                        initargs=(
                            self.seg_model,
                            self.text_key,
                            self.split_chinese,
                            self.max_text_length,
                            self.skip_empty,
                        )
                    ) as pool:
                        # 遍历文件的每个块
                        for chunk in self._chunk_generator(file_path):
                            # 使用 imap 并行处理块中的每一行
                            # 注意：chunksize=1 表示每个任务单独调度，因为 chunk 本身可能很大
                            for text in pool.imap(_process_line_worker, chunk, chunksize=1):
                                if text is not None:
                                    yield text
                else:
                    # 单进程流式逐行模式
                    yield from self._file_generator_single(file_path)