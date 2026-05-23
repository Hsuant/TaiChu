#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试数据集加载器（TaiChuPretrainDataset 和 build_dataloader）。

测试内容：
- 数据集长度、样本格式、BOS/EOS 标记是否正确
- 文本截断、填充、padding token 的标签屏蔽
- 空文本处理
- DataLoader 构建：单进程与分布式采样器
- 多 worker 数据加载的兼容性
"""

import unittest
import tempfile
import json
import os
from types import SimpleNamespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.llm_dataset import TaiChuPretrainDataset, build_dataloader  # type: ignore


class MockTokenizer:
    """模拟 tokenizer，用于测试数据集而不依赖真实 tokenizer 文件。

    支持基本功能：
        - encode(text): 返回一个包含 ids 列表的简单对象（类似 tokenizers.Encoding）
        - token_to_id(token): 返回特殊 token 对应的 ID，若不存在则抛出 KeyError
    """

    def __init__(self):
        # 定义特殊 token 及其 ID（硬编码）
        self.special_tokens: Dict[str, int] = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,     # EOS 同时也是 PAD
        }
        # 普通字符到 ID 的映射（简化处理，只支持字母和空格）
        self.char_to_id: Dict[str, int] = {chr(i): i + 10 for i in range(32, 127)}
        self.char_to_id[' '] = 4

    def token_to_id(self, token: str) -> int:
        """返回特殊 token 的 ID，如果不存在则抛出 KeyError（模仿真实 tokenizer）。"""
        if token not in self.special_tokens:
            raise KeyError(f"Unknown token: {token}")
        return self.special_tokens[token]

    def encode(self, text: str):
        """模拟编码，将每个字符转换为固定 ID，返回一个带 ids 属性的对象。"""
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                # 中文字符或其它，统一映射到 1000
                ids.append(1000)

        # 模仿 tokenizers.Encoding 对象，只需提供 ids 属性
        class Encoding:
            def __init__(self, ids: List[int]):
                self.ids = ids

        return Encoding(ids)

    def __call__(self, text: str, *args, **kwargs):
        """允许直接调用 tokenizer 对象（兼容原始接口）。"""
        return self.encode(text)


def create_temp_json_file(lines: List[str]) -> str:
    """创建临时 JSON 文件（每行一个 JSON 对象）。"""
    fd, path = tempfile.mkstemp(suffix='.json', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for line in lines:
            json.dump({"text": line}, f, ensure_ascii=False)
            f.write('\n')
    return path


class TestTaiChuPretrainDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.max_length = 10
        self.text_field = "text"

        # 注意：索引顺序
        # 0: "你好世界" (4个中文字符)
        # 1: "Hello, world!" (13个字符)
        # 2: "" (空字符串)
        # 3: "这是一个较长的句子，用来测试截断功能。" (长文本)
        self.lines = [
            "你好世界",
            "Hello, world!",
            "",
            "这是一个较长的句子，用来测试截断功能。",
        ]
        self.data_path = create_temp_json_file(self.lines)

    def tearDown(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

    def test_dataset_length(self):
        dataset = TaiChuPretrainDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,  # type: ignore
            max_length=self.max_length,
            text_field=self.text_field,
        )
        self.assertEqual(len(dataset), len(self.lines))

    def test_sample_format(self):
        dataset = TaiChuPretrainDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,  # type: ignore
            max_length=self.max_length,
            text_field=self.text_field,
        )
        sample = dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("labels", sample)
        self.assertIsInstance(sample["input_ids"], torch.Tensor)
        self.assertIsInstance(sample["labels"], torch.Tensor)
        self.assertEqual(sample["input_ids"].shape, (self.max_length,))
        self.assertEqual(sample["labels"].shape, (self.max_length,))

    def test_bos_eos_padding(self):
        """测试正常文本的 BOS/EOS 和填充。

        "你好世界" 有 4 个字符，每个映射到 1000，
        序列应为 [BOS(1), 1000, 1000, 1000, 1000, EOS(2)] + 填充(3)
        max_length=10，填充 4 个 PAD。
        """
        dataset = TaiChuPretrainDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,  # type: ignore
            max_length=self.max_length,
            text_field=self.text_field,
        )
        sample = dataset[0]  # "你好世界"
        input_ids = sample["input_ids"].tolist()
        labels = sample["labels"].tolist()

        # 预期：4个1000，因为4个中文字符
        # 序列：[BOS(1)] + 4个1000 + [EOS/PAD(2)] + 填充(2)重复
        expected_input = [1, 1000, 1000, 1000, 1000, 2] + [2] * (self.max_length - 6)
        self.assertEqual(input_ids, expected_input)

        # 标签：BOS和内容保持原值，EOS/PAD及填充位置设为 -100
        expected_labels = [1, 1000, 1000, 1000, 1000, -100] + [-100] * (self.max_length - 6)
        self.assertEqual(labels, expected_labels)

    def test_truncation(self):
        """测试文本截断：max_length=5，有效内容长度=3，
        应取长文本的前3个 token + BOS/EOS，无填充。
        """
        short_max_len = 5
        dataset = TaiChuPretrainDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,  # type: ignore
            max_length=short_max_len,
            text_field=self.text_field,
        )
        # 索引 3 是长文本
        sample = dataset[3]
        input_ids = sample["input_ids"].tolist()
        self.assertEqual(len(input_ids), short_max_len)
        self.assertEqual(input_ids[0], 1)  # BOS
        self.assertEqual(input_ids[-1], 2)  # EOS
        # 中间3个应该是文本的前3个字符（中文字符映射为1000）
        self.assertEqual(input_ids[1:4], [1000, 1000, 1000])

    def test_empty_text(self):
        """空文本：原数据集将空字符串替换为空格，但 strip() 后变成空，
        因此实际编码为空，序列为 BOS + EOS + 填充。
        """
        dataset = TaiChuPretrainDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,  # type: ignore
            max_length=self.max_length,
            text_field=self.text_field,
        )
        # 索引 2 是空字符串
        sample = dataset[2]
        input_ids = sample["input_ids"].tolist()
        # 预期：BOS(1), EOS(2), 然后全部填充(3)
        expected = [1, 2] + [2] * (self.max_length - 2)
        self.assertEqual(input_ids, expected)

    def test_labels_ignore_padding(self):
        dataset = TaiChuPretrainDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,  # type: ignore
            max_length=self.max_length,
            text_field=self.text_field,
        )
        sample = dataset[0]
        labels = sample["labels"].tolist()
        for i, label in enumerate(labels):
            if sample["input_ids"][i] == 2:  # PAD token id
                self.assertEqual(label, -100)
            else:
                self.assertNotEqual(label, -100)

    def test_invalid_text_field(self):
        """JSON 中缺少 text 字段，回退为空字符串，行为同空文本。"""
        fd, path = tempfile.mkstemp(suffix='.json', text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({"other_field": "some text"}, f)
            f.write('\n')
        dataset = TaiChuPretrainDataset(
            data_path=path,
            tokenizer=self.tokenizer,  # type: ignore
            max_length=self.max_length,
            text_field="text",
        )
        sample = dataset[0]
        input_ids = sample["input_ids"].tolist()
        # 同空文本：BOS + EOS + 填充
        expected = [1, 2] + [2] * (self.max_length - 2)
        self.assertEqual(input_ids, expected)
        os.remove(path)


class TestBuildDataloader(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.lines = [f"这是第{i}条测试文本" for i in range(20)]
        self.data_path = create_temp_json_file(self.lines)

        self.data_config = SimpleNamespace()
        self.data_config.train_path = self.data_path
        self.data_config.val_path = self.data_path
        self.data_config.test_path = self.data_path
        self.data_config.max_length = 16
        self.data_config.text_field = "text"
        self.data_config.num_workers = 0

    def tearDown(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

    def test_build_dataloader_train(self):
        dataloader = build_dataloader(
            data_config=self.data_config,
            tokenizer=self.tokenizer,  # type: ignore
            split="train",
            batch_size=4,
            rank=0,
            world_size=1,
        )
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(dataloader.batch_size, 4)
        self.assertTrue(dataloader.drop_last)

        for batch in dataloader:
            self.assertIn("input_ids", batch)
            self.assertIn("labels", batch)
            self.assertEqual(batch["input_ids"].shape, (4, self.data_config.max_length))
            self.assertEqual(batch["labels"].shape, (4, self.data_config.max_length))
            break

    def test_build_dataloader_val(self):
        dataloader = build_dataloader(
            data_config=self.data_config,
            tokenizer=self.tokenizer,  # type: ignore
            split="val",
            batch_size=8,
            rank=0,
            world_size=1,
        )
        self.assertFalse(dataloader.drop_last)

    def test_build_dataloader_test(self):
        dataloader = build_dataloader(
            data_config=self.data_config,
            tokenizer=self.tokenizer,  # type: ignore
            split="test",
            batch_size=8,
        )
        self.assertFalse(dataloader.drop_last)

    def test_distributed_sampler(self):
        dataloader = build_dataloader(
            data_config=self.data_config,
            tokenizer=self.tokenizer,  # type: ignore
            split="train",
            batch_size=2,
            rank=0,
            world_size=2,
        )
        self.assertIsNotNone(dataloader.sampler)
        self.assertIsInstance(dataloader.sampler, DistributedSampler)
        self.assertTrue(dataloader.drop_last)

    def test_invalid_split(self):
        with self.assertRaises(ValueError):
            build_dataloader(
                data_config=self.data_config,
                tokenizer=self.tokenizer,  # type: ignore
                split="invalid",
                batch_size=4,
            )

    def test_missing_data_path(self):
        config = SimpleNamespace()
        config.train_path = None
        with self.assertRaises(ValueError):
            build_dataloader(
                data_config=config,
                tokenizer=self.tokenizer,  # type: ignore
                split="train",
                batch_size=4,
            )

    def test_num_workers_fallback(self):
        config = SimpleNamespace()
        config.train_path = self.data_path
        config.num_workers = 4
        dataloader = build_dataloader(
            data_config=config,
            tokenizer=self.tokenizer,  # type: ignore
            split="train",
            batch_size=4,
            num_workers=2,
        )
        self.assertEqual(dataloader.num_workers, 2)

        dataloader2 = build_dataloader(
            data_config=config,
            tokenizer=self.tokenizer,  # type: ignore
            split="train",
            batch_size=4,
        )
        self.assertEqual(dataloader2.num_workers, 4)


if __name__ == '__main__':
    unittest.main()