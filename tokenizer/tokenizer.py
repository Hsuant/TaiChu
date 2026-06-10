"""
TaiChu Tokenizer 训练主控制器。

负责将数据流、规范化器、预分词器、训练器、后处理器和解码器组装起来，
执行完整的 Byte-Level BPE 训练流程，并保存为 HuggingFace 兼容格式。

生成的文件：
    - tokenizer.json          : 完整的分词器定义（模型、词表等）
    - vocab.json / merges.txt : BPE 词表与合并规则（由 save_pretrained 生成）
    - tokenizer_config.json   : HuggingFace Transformers 所需配置（定制）
    - special_tokens_map.json : 特殊 token 映射
"""

import os
import json
from typing import Any, Dict

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from tokenizer.corpus import CorpusIterator
from tokenizer.normalizer import NormalizerFactory
from tokenizer.pre_tokenizer import PreTokenizerFactory
from tokenizer.decoder import DecoderFactory


class TrainerFactory:
    """TaiChuTokenizer 训练器工厂，负责按配置创建 BpeTrainer 实例。
    """

    @staticmethod
    def create(config: Dict[str, Any]) -> BpeTrainer:
        """根据配置字典创建 BpeTrainer。

        Args:
            config: 字典，可包含以下键：
                - vocab_size: 词汇表大小，默认 128000。
                - min_frequency: 最小合并频次，默认 0（保留所有 pair）。
                - show_progress: 是否显示进度条，默认 True。
                - special_tokens: 特殊 token 列表，必须在训练前指定。
                - initial_alphabet: 初始字母表（通常由 ByteLevel 提供）。

        Returns:
            配置好的 BpeTrainer 实例。
        """
        vocab_size = config.get("vocab_size", 128000)
        min_frequency = config.get("min_frequency", 1)
        show_progress = config.get("show_progress", True)
        special_tokens = config.get("special_tokens", [])

        return BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
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
        # ---------- 0. 提前注册特殊 token（确保 ID 连续靠前） ----------
        trainer_config = self.config['trainer']
        all_special_tokens = trainer_config.get("special_tokens", [])
        if all_special_tokens:
            # AddedToken 设置 normalized=False 防止规范化器修改，
            # 保证特殊 token 的字符完整性。
            added_tokens = [
                AddedToken(token, normalized=False, lstrip=False, rstrip=False)
                for token in all_special_tokens
            ]
            self.tokenizer.add_special_tokens(added_tokens)
            print(
                f"训练前已注册 {len(all_special_tokens)} 个特殊 token，"
                f"ID 范围: {self.tokenizer.token_to_id(all_special_tokens[0])} "
                f"~ {self.tokenizer.token_to_id(all_special_tokens[-1])}"
            )

        # ---------- 1. 准备数据 ----------
        data_config = self.config['data']
        corpus = CorpusIterator(
            files=data_config['files'],
            epoch=data_config.get('epoch', 1),
            text_key=data_config.get('text_key', 'text'),
            max_text_length=data_config.get('max_text_length', None),
            num_workers=data_config.get('num_workers', 0),
            skip_empty=data_config.get('skip_empty', True),
            chunk_size = data_config.get('chunk_size', 5000)
        )

        # ---------- 2. 设置预分词器 ----------
        pre_config = self.config.get('pre_tokenizer', {})
        self.tokenizer.pre_tokenizer = PreTokenizerFactory.create(pre_config)

        # ---------- 3. 配置训练器 ----------
        trainer_config = self.config['trainer']
        trainer = TrainerFactory.create(trainer_config)

        # ---------- 4. 开始训练 ----------
        print(
            f"开始训练 BPE 分词器，目标词表大小: "
            f"{trainer_config.get('vocab_size')}，最小词频: "
            f"{trainer_config.get('min_frequency', 5)}..."
        )
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

        # ------- 1. 写入 chat_template（Jinja2 模板） -------
        chat_template = self.config.get('chat_template', None)
        if chat_template:
            self.tokenizer.chat_template = chat_template
            print("已写入自定义 chat_template。")

        # ------- 2. 分配标准特殊 token 属性 -------
        self._assign_special_tokens()

        # ------- 3. 保存主模型文件 tokenizer.json -------
        save_path = os.path.join(save_dir, "tokenizer.json")
        self.tokenizer.save(save_path)
        print(f"分词器已保存至: {save_path}")

        # ------- 4. 导出 HuggingFace 完整兼容包 -------
        self._export_huggingface_compatibility(save_dir)

    def _assign_special_tokens(self):
        """为 tokenizer 对象设置 bos/eos/unk/pad 等标准特殊 token 属性。

        这些属性是 HuggingFace Transformers 识别特殊标记的关键。
        pad_token 复用 eos_token，避免额外占用词表空间。
        """
        bos_token = "<|im_start|>"  # 同时作为 BOS 和消息起始标记
        eos_token = "<|im_end|>"  # 消息结束标记
        unk_token = "<unk>"  # 未登录词标记
        pad_token = eos_token  # 复用 eos 作为 pad，符合常见实践

        # 设置属性时，tokenizers 库会自动查找或添加对应的 token ID
        self.tokenizer.bos_token = bos_token
        self.tokenizer.eos_token = eos_token
        self.tokenizer.unk_token = unk_token
        self.tokenizer.pad_token = pad_token

    def _export_huggingface_compatibility(self, save_dir: str):
        """导出完整的 HuggingFace Transformers 兼容文件包。

        执行步骤：
            1. 通过 PreTrainedTokenizerFast 保存标准模型文件，
               自动生成 vocab.json、merges.txt 等。
            2. 用自定义配置覆盖 tokenizer_config.json，加入
               chat_template、model_max_length 等扩展信息。
            3. 定制 special_tokens_map.json，精确分离 standard tokens
               与 additional_special_tokens，满足下游框架要求。

        Args:
            save_dir: 输出目录路径。
        """
        from transformers import PreTrainedTokenizerFast

        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        unk_token = self.tokenizer.unk_token
        pad_token = self.tokenizer.pad_token

        # 1. 调用 HuggingFace 标准保存，生成基础文件
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )
        fast_tokenizer.save_pretrained(save_dir)
        print(f"HuggingFace 基础包已保存至: {save_dir}")

        # 2. 定制 tokenizer_config.json（添加 chat_template、max_length 等）
        chat_template = self.config.get('chat_template', '')
        model_meta = self.config.get("model_meta", None)
        hf_config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": False,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "unk_token": unk_token,
            # 极大值表示无最大长度限制，实际限制由模型架构决定
            "model_max_length": 1000000000000000019884624838656,
            "clean_up_tokenization_spaces": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "chat_template": chat_template if chat_template else "",
        }
        # 可选：添加专属模型配置信息
        if model_meta:
            hf_config["model_info"] = model_meta
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(hf_config, f, ensure_ascii=False, indent=4)

        # 3. 定制 special_tokens_map.json
        trainer_config = self.config["trainer"]
        all_special_tokens = trainer_config.get("special_tokens", [])

        # 标准 token 已通过 bos/eos/unk/pad 属性设置
        standard_tokens = {bos_token, eos_token, unk_token, pad_token}
        # 附加特殊 token 为 all_special_tokens 中除去标准 token 后的剩余部分
        additional_special_tokens = [
            t for t in all_special_tokens if t not in standard_tokens
        ]

        special_tokens_map = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "additional_special_tokens": additional_special_tokens,
        }
        with open(os.path.join(save_dir, "special_tokens_map.json"), "w",
                  encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

        # 4. 日志输出：关键信息供核查
        print("已生成 HuggingFace 兼容配置文件。")
        print(f"词表大小: {self.tokenizer.get_vocab_size()}")
        special_tokens = trainer_config.get("special_tokens", [])
        for token in special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            print(f"特殊 Token: '{token}' -> ID: {token_id}")