"""
预分词器工厂模块。根据配置文件创建不同类型的预分词器。
字节级（ByteLevel）是BPE常用且鲁棒的预分词方式，被Llama等模型采用。
"""
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, CharDelimiterSplit


class PreTokenizerFactory:
    """预分词器工厂，根据类型返回对应的预分词器实例。"""

    @staticmethod
    def create(pre_tokenizer_config):
        """
        创建预分词器实例。

        Args:
            pre_tokenizer_config (dict): 预分词器配置。

        Returns:
            PreTokenizer: 预分词器实例。

        Raises:
            ValueError: 当遇到不支持的预分词器类型时抛出。
        """
        p_type = pre_tokenizer_config.get("type", "bytelevel")
        if p_type == "bytelevel":
            return ByteLevel(add_prefix_space=False, use_regex=True)
        elif p_type == "whitespace":
            return Whitespace()
        else:
            raise ValueError(f"不支持的预分词器类型: {p_type}")