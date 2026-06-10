# -*- coding: utf-8 -*-
"""数据集迭代器模块。

该模块提供 CorpusIterator 类，用于从 JSONL 格式文件中流式读取文本数据，
支持可选的并行处理（多进程）。
主要用于 BPE 分词器训练前的语料预处理。

优化点：
    - 避免一次性读入整个文件，采用逐块（chunk）读取，显著降低主进程内存峰值。
    - 多进程模式下仍保持流式处理，通过 imap 逐块提交任务，内存开销稳定。
    - 子进程全局变量存储静态配置，单任务仅传递一行字符串，IPC 负载极小。
    - 进程池全局复用，避免重复创建/销毁。
"""

import json
from multiprocessing import Pool, cpu_count
from typing import Iterator, List, Optional

# ==================== 多进程全局变量 ====================
_worker_text_key = None
_worker_max_text_length = None
_worker_skip_empty = None


def _init_worker(text_key: str, max_text_length: Optional[int], skip_empty: bool) -> None:
    """在每个子进程启动时调用，初始化全局配置。

    Args:
        text_key: JSON 对象中文本字段的键名。
        max_text_length: 单条文本的最大长度限制（字符数）。
        skip_empty: 是否跳过空文本或仅含空白字符的文本。
    """
    global _worker_text_key, _worker_max_text_length, _worker_skip_empty
    _worker_text_key = text_key
    _worker_max_text_length = max_text_length
    _worker_skip_empty = skip_empty


def _process_line_worker(line: str) -> Optional[str]:
    """供多进程调用的处理函数，使用全局配置处理单行 JSON 数据。

    Args:
        line: 原始字符串行，应为 JSON 格式。

    Returns:
        处理后的文本字符串，若数据无效则返回 None。
    """
    global _worker_text_key, _worker_max_text_length, _worker_skip_empty

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

    return text


class CorpusIterator:
    """可迭代的语料库，逐条生成训练文本。

    该类采用**流式分块**策略：
        - 单进程模式：逐行读取并 yield，内存占用极小。
        - 多进程模式：按行块（chunk）读取，每块中的每一行单独提交给进程池，
          主进程仅保留当前块，内存峰值稳定。

    Attributes:
        files: JSONL 文件路径列表。
        epoch: 遍历整个数据集的次数（轮数）。
        text_key: JSON 对象中文本字段的名称。
        max_text_length: 单条文本允许的最大字符数。
        num_workers: 并行处理的进程数，0 表示自动使用 CPU 核心数。
        skip_empty: 是否跳过空文本或仅含空白字符的文本。
        chunk_size: 多进程模式下每个任务包含的行数（块大小）。
    """

    def __init__(
        self,
        files: List[str],
        epoch: int = 1,
        text_key: str = "text",
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
            max_text_length: 最大文本长度（字符数），None 表示不限制。
            num_workers: 并行进程数，0 表示自动使用 CPU 核心数。
            skip_empty: 是否跳过空文本或仅含空白字符的文本。
            chunk_size: 多进程模式下每个任务包含的行数（块大小）。
        """
        self.files = files
        self.epoch = epoch
        self.text_key = text_key
        self.max_text_length = max_text_length
        self.num_workers = num_workers if num_workers > 0 else cpu_count()
        self.skip_empty = skip_empty
        self.chunk_size = chunk_size

    def _file_generator_single(self, file_path: str) -> Iterator[str]:
        """单进程模式：顺序读取单个文件并生成处理后的文本。

        Args:
            file_path: JSONL 文件路径。

        Yields:
            经过清洗后的文本字符串。
        """
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
            - 每个块通过 pool.imap 提交给工作进程并行处理，chunksize=1 确保每个任务仅一行。
            - imap 返回的结果流式产出，不会一次性加载所有结果。
            - 进程池在外部创建一次，跨所有文件和 epoch 复用。

        Yields:
            经过清洗后的文本字符串。
        """
        # 多进程模式
        if self.num_workers > 1:
            # 进程池全局复用，避免重复创建销毁
            with Pool(
                self.num_workers,
                initializer=_init_worker,
                initargs=(self.text_key, self.max_text_length, self.skip_empty),
            ) as pool:
                for _ in range(self.epoch):
                    for file_path in self.files:
                        for chunk in self._chunk_generator(file_path):
                            # 每个任务只传一行，配置已固化在子进程全局变量中
                            for text in pool.imap_unordered(_process_line_worker, chunk, chunksize=1):
                                if text is not None:
                                    yield text
        # 单进程模式
        else:
            for _ in range(self.epoch):
                for file_path in self.files:
                    yield from self._file_generator_single(file_path)