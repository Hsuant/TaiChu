"""
全量加载的预训练数据集（基于 HuggingFace datasets）。

适用于小规模 JSON 格式语料（< 10 GB），一次性将数据索引加载到内存。
大规模预训练建议使用流式 IterableDataset。
"""

from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tokenizers import Tokenizer

from typing import Optional, Union, List


class TaiChuPretrainDataset(Dataset):
    """全量加载的预训练数据集。

    使用 datasets 库加载 JSON 文件，将每条文本截断到固定长度，
    并在前后添加 BOS 和 EOS 标记。未填满的位置用 PAD 填充，
    其标签设为 -100 以忽略损失。

    Attributes:
        samples: datasets 库返回的表格数据集。
        tokenizer: 已训练的 tokenizer 实例。
        max_length: 序列最大长度（含 BOS/EOS）。
        text_field: JSON 对象中文本字段的名称。
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: Tokenizer,
        max_length: int = 2048,
        text_field: str = "text",
    ):
        """初始化数据集。

        Args:
            data_files: JSON/JSONL 文件路径，支持通配符或文件列表。
            tokenizer: 已训练的 TaiChu tokenizer 实例。
            max_length: 序列最大长度（包含 BOS 和 EOS）。
            text_field: JSON 中文本字段的键名。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field

        # 加载 JSON 数据(支持单个文件路径、通配符或文件列表）
        self.samples = load_dataset('json', data_files=data_files, split='train')

        # 获取特殊 token 的 ID（根据实际使用的 token 名称调整）
        self.bos_token_id = tokenizer.token_to_id("<|im_start|>")  # BOS 标记
        self.eos_token_id = tokenizer.token_to_id("<|im_end|>")  # EOS 标记
        self.pad_token_id = tokenizer.token_to_id("<|im_end|>")  # PAD 标记

        # 确保 tokenizer 已设置必要的特殊 token ID
        assert self.bos_token_id is not None, "tokenizer 缺少 bos_token_id"
        assert self.eos_token_id is not None, "tokenizer 缺少 eos_token_id"
        assert self.pad_token_id is not None, "tokenizer 缺少 pad_token_id"

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """获取单个样本。

        Args:
            index: 样本索引。

        Returns:
            包含 "input_ids" 和 "labels" 的字典，均为 torch.LongTensor。
        """
        sample = self.samples[index]
        # 提取文本，并转换为字符串（对非字符串字段安全）
        text = str(sample.get(self.text_field, ""))
        if not text:
            # 若为空文本，用一个空格代替，防止编码出错
            text = " "

        # 使用 tokenizer 编码（add_special_tokens=False 因为我们手动添加）
        encoding = self.tokenizer.encode(text.strip())
        token_ids = encoding.ids

        # 预留 BOS 和 EOS 的位置
        content_max_len = self.max_length - 2
        if len(token_ids) > content_max_len:
            token_ids = token_ids[:content_max_len]

        # 构造完整序列：[BOS] + content + [EOS] + [PAD]...
        input_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        # 填充到 max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.pad_token_id] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # 忽略填充位置的损失
        labels[input_ids == self.pad_token_id] = -100

        return {"input_ids": input_ids, "labels": labels}


def build_dataloader(
    data_config,
    tokenizer: Tokenizer,
    split: str,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """构建支持训练、验证、测试的数据加载器。

    支持从配置对象中读取文件列表（train_files / val_files / test_files），
    同时也兼容旧版单一文件路径（train_path / val_path / test_path）。

    Args:
        data_config: 配置对象，需包含以下属性：
            - train_files / val_files / test_files: 对应 split 的数据文件列表
            - max_length: 序列最大长度
            - text_field: 文本字段名，默认为 "text"
            - num_workers: 数据加载进程数（可选）
        tokenizer: 已训练的 tokenizer 实例。
        split: 数据集类型，支持 "train", "val", "test"。
        batch_size: 批次大小（每个设备的样本数）。
        rank: 当前进程的全局 rank（用于分布式采样）。
        world_size: 总进程数。
        shuffle: 是否打乱数据。若为 None，则训练集自动为 True，验证/测试集为 False。
        num_workers: 数据加载子进程数，若 data_config 中有则优先使用传入值。
        pin_memory: 是否锁定内存（加速 GPU 传输）。

    Returns:
        DataLoader 实例。

    Raises:
        ValueError: 当 split 不支持或对应路径缺失时。
    """
    # 确定数据文件列表
    if split == "train":
        data_files = data_config.train_files
        default_shuffle = True
    elif split == "val":
        data_files = data_config.val_files
        default_shuffle = False
    elif split == "test":
        data_files = data_config.test_files
        default_shuffle = False
    else:
        raise ValueError(f"不支持的 split: {split}，仅支持 'train', 'val', 'test'")

    # 获取数据集参数
    max_length = getattr(data_config, "max_length", 2048)
    text_field = getattr(data_config, "text_field", "text")

    # 确定 num_workers 优先级：传入参数 > data_config.num_workers
    if num_workers == 0 and hasattr(data_config, "num_workers"):
        num_workers = data_config.num_workers

    # 创建数据集
    dataset = TaiChuPretrainDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        max_length=max_length,
        text_field=text_field,
    )

    # 确定是否打乱数据
    if shuffle is None:
        shuffle = default_shuffle

    # 分布式采样器（只在多卡且训练集时使用，验证集也可用但通常不用）
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=(split == "train"),
        )
        # 使用 sampler 时必须将 shuffle 设为 False，避免重复打乱
        shuffle = False

    # 创建 DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),  # 训练时丢弃最后一个不完整的 batch
    )