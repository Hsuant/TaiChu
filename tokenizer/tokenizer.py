"""
TaiChu Tokenizer 训练主控制器。

负责将数据流、规范化器、预分词器、训练器、后处理器和解码器组装起来，
执行完整的 Byte-Level BPE 训练流程，并保存为 HuggingFace 兼容格式。
"""

import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizer.corpus import CorpusIterator
from tokenizer.normalizer import NormalizerFactory
from tokenizer.pre_tokenizer import PreTokenizerFactory
from tokenizer.decoder import DecoderFactory


class TrainerFactory:
    """训练器工厂，负责配置并创建 BpeTrainer 实例。"""

    @staticmethod
    def create(trainer_config):
        """根据配置字典创建 BpeTrainer。

        Args:
            trainer_config (dict): BPE 训练器的配置，包括：
                vocab_size, min_frequency, show_progress, special_tokens 等。

        Returns:
            BpeTrainer: 配置好的训练器实例。
        """
        return BpeTrainer(
            vocab_size=trainer_config.get("vocab_size", 64000),
            min_frequency=trainer_config.get("min_frequency", 2),
            show_progress=trainer_config.get("show_progress", True),
            special_tokens=trainer_config.get("special_tokens", []),
            initial_alphabet=ByteLevel.alphabet()  # 保证覆盖所有 256 个字节
        )


class TaiChuTokenizerTrainer:
    """负责整个 BPE Tokenizer 的训练流程。

    Attributes:
        config: 从 YAML 文件加载的完整配置字典。
        tokenizer: 待训练的 Tokenizer 对象。
    """

    def __init__(self, config):
        """初始化训练器，设置模型与规范化器。

        Args:
            config: 配置字典。
        """
        self.config = config
        # 初始化 BPE 模型，unk_token 为 "<unk>"
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        # 根据配置设置规范化器
        norm_config = config.get('normalizer', {})
        self.tokenizer.normalizer = NormalizerFactory.create(norm_config)

    def run(self):
        """执行完整的训练、后处理、解码器配置和保存流程。"""
        # ---------- 1. 准备数据 ----------
        data_config = self.config['data']
        corpus = CorpusIterator(
            files=data_config['files'],
            epoch=data_config.get('epoch', 1),
            text_key=data_config.get('text_key', 'text'),
            split_chinese=data_config.get('split_chinese', False),
            seg_model=data_config.get('seg_model', 'mixed'),
            max_text_length=data_config.get('max_text_length', None),
            num_workers=data_config.get('num_workers', 0),
            skip_empty=data_config.get('skip_empty', True)
        )

        # ---------- 2. 设置预分词器 ----------
        pre_config = self.config.get('pre_tokenizer', {})
        self.tokenizer.pre_tokenizer = PreTokenizerFactory.create(pre_config)

        # ---------- 3. 配置训练器 ----------
        trainer_config = self.config['trainer']
        trainer = TrainerFactory.create(trainer_config)

        # ---------- 4. 开始训练 ----------
        print(f"开始训练 BPE 分词器，目标词表大小: {trainer_config.get('vocab_size')}...")
        self.tokenizer.train_from_iterator(corpus, trainer)
        print("训练完成。")

        # ---------- 5. 设置解码器 ----------
        decoder_config = self.config['decoder']
        self.tokenizer.decoder = DecoderFactory.create(decoder_config)

        # ---------- 6. 保存分词器及相关文件 ----------
        self._save_tokenizer()

    def _save_tokenizer(self):
        """将训练好的 tokenizer 保存到磁盘，并生成 HuggingFace 兼容配置。"""
        save_config = self.config['save']
        save_dir = save_config['directory']
        os.makedirs(save_dir, exist_ok=True)

        # 写入 chat_template（Jinja2 模板）
        chat_template = self.config.get('chat_template', None)
        if chat_template:
            self.tokenizer.chat_template = chat_template
            print("已写入自定义 chat_template。")

        # 设置特殊 token 属性
        bos_token = "<|im_start|>"  # 同时作为 BOS 和消息起始标记
        eos_token = "<|im_end|>"    # 消息结束标记
        unk_token = "<unk>"         # 未登录词标记
        pad_token = eos_token       # 复用 eos 作为 pad

        self.tokenizer.bos_token = bos_token
        self.tokenizer.eos_token = eos_token
        self.tokenizer.unk_token = unk_token
        self.tokenizer.pad_token = pad_token
        # 自动计算并设置 token_id（tokenizers 库支持）

        # 保存主模型文件
        save_path = os.path.join(save_dir, "tokenizer.json")
        self.tokenizer.save(save_path)
        print(f"分词器已保存至: {save_path}")

        # 生成 HuggingFace transformers 兼容的 tokenizer_config.json
        hf_config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": False,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "unk_token": unk_token,
            "model_max_length": 1000000000000000019884624838656,  # 极大值，表示无限制
            "clean_up_tokenization_spaces": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "chat_template": chat_template if chat_template else "",
        }
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(hf_config, f, ensure_ascii=False, indent=4)

        # 生成 special_tokens_map.json
        special_tokens_map = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "additional_special_tokens": [
                "<|think|>",
                "<|/think|>",
                "<|endoftext|>",
            ]
        }
        with open(os.path.join(save_dir, "special_tokens_map.json"), "w",
                  encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

        print("已生成 HuggingFace 兼容配置文件。")
        print(f"词表大小: {self.tokenizer.get_vocab_size()}")
        special_tokens = self.config['trainer'].get('special_tokens', [])
        for token in special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            print(f"特殊 Token: '{token}' -> ID: {token_id}")