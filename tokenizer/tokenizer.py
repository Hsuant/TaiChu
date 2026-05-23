"""
TaiChu Tokenizer 训练主控制器。
负责将数据流、预分词器、训练器、后处理器和解码器组装起来，执行完整的训练流程。
"""
import os
import json
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from .corpus import CorpusIterator
from .pre_tokenizer import PreTokenizerFactory
from .post_processor import PostProcessorFactory
from .decoder import DecoderFactory


class TrainerFactory:
    """训练器工厂，负责配置并创建BpeTrainer实例。"""

    @staticmethod
    def create(trainer_config):
        """
        根据配置创建BpeTrainer实例。

        Args:
            trainer_config (dict): BPE训练器的配置。

        Returns:
            BpeTrainer: 配置好的训练器实例。
        """
        return BpeTrainer(
            vocab_size=trainer_config.get("vocab_size", 50257),
            min_frequency=trainer_config.get("min_frequency", 2),  # 低频词过滤，可减少词表和内存
            limit_alphabet=trainer_config.get("limit_alphabet", 1000),  # 限制初始字母表，控制内存峰值
            show_progress=trainer_config.get("show_progress", True),
            special_tokens=trainer_config.get("special_tokens", []),
            initial_alphabet=ByteLevel.alphabet()  # 确保完整覆盖所有字节
        )


class TaiChuTokenizerTrainer:
    """
    负责整个BPE Tokenizer的训练流程。

    Args:
        config (dict): 从YAML文件中加载的完整配置。
    """

    def __init__(self, config):
        self.config = config
        # 使用字节级BPE模型进行初始化
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        # 设置 NFKC 标准化器
        self.tokenizer.normalizer = normalizers.NFKC()

    def run(self):
        """执行完整的训练、后处理、解码器配置和保存流程。"""
        # 1. 准备数据
        data_config = self.config['data']
        corpus = CorpusIterator(data_config['files'], data_config.get('epoch', 1))

        # 2. 设置预分词器
        pre_tokenizer_config = self.config.get('pre_tokenizer', {})
        self.tokenizer.pre_tokenizer = PreTokenizerFactory.create(pre_tokenizer_config)

        # 3. 配置训练器
        trainer_config = self.config['trainer']
        trainer = TrainerFactory.create(trainer_config)

        # 4. 开始训练
        print(f"开始训练BPE分词器，目标词表大小: {trainer_config.get('vocab_size')}...")
        self.tokenizer.train_from_iterator(corpus, trainer)
        print("训练完成。")

        # 5. 设置后处理器（如对话模板）
        post_processor_config = self.config.get('post_processor', {})
        self.tokenizer.post_processor = PostProcessorFactory.create(post_processor_config, self.tokenizer)

        # 6. 设置解码器
        decoder_config = self.config.get('decoder', {})
        self.tokenizer.decoder = DecoderFactory.create(decoder_config)

        # 7. 保存分词器
        self._save_tokenizer()

    def _save_tokenizer(self):
        """保存训练好的分词器到指定路径。"""
        save_config = self.config['save']
        save_dir = save_config['directory']
        os.makedirs(save_dir, exist_ok=True)

        # 写入 chat_template
        chat_template = self.config.get('chat_template', None)
        if chat_template:
            self.tokenizer.chat_template = chat_template
            print("已写入自定义 chat_template。")

        # 设置特殊 token 属性
        # 根据我们词表中的实际特殊 token 定义
        bos_token = "<|im_start|>"
        eos_token = "<|im_end|>"
        unk_token = "<unk>"
        pad_token = eos_token  # 复用 eos 作为 pad

        # 设置 tokenizer 对象属性（tokenizers 库支持）
        self.tokenizer.bos_token = bos_token
        self.tokenizer.bos_token_id = self.tokenizer.token_to_id(bos_token)
        self.tokenizer.eos_token = eos_token
        self.tokenizer.eos_token_id = self.tokenizer.token_to_id(eos_token)
        self.tokenizer.unk_token = unk_token
        self.tokenizer.unk_token_id = self.tokenizer.token_to_id(unk_token)
        self.tokenizer.pad_token = pad_token
        self.tokenizer.pad_token_id = self.tokenizer.token_to_id(pad_token)

        save_path = os.path.join(save_dir, "tokenizer.json")
        self.tokenizer.save(save_path)
        print(f"分词器已保存至: {save_path}")

        # 导出 HuggingFace transformers 兼容的配置文件
        # 1. tokenizer_config.json
        hf_config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": True,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "unk_token": unk_token,
            "model_max_length": 1000000000000000019884624838656,  # 极大值，表示无限制
            "clean_up_tokenization_spaces": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "chat_template": chat_template if chat_template else "",
        }
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_config, f, ensure_ascii=False, indent=4)

        # 2. special_tokens_map.json
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
        with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)
        print("已生成 HuggingFace 兼容配置文件。")

        # 为方便调试，打印词表大小和特殊token的ID
        print(f"词表大小: {self.tokenizer.get_vocab_size()}")
        special_tokens = self.config['trainer'].get('special_tokens', [])
        for token in special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            print(f"特殊Token: '{token}' -> ID: {token_id}")