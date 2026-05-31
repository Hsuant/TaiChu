"""
解码器工厂模块。

根据配置创建解码器实例，用于将 Token ID 序列转换回原始文本。
目前仅支持 ByteLevel 解码器。
"""
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class DecoderFactory:
    """解码器工厂，根据类型返回对应的解码器实例。"""

    @staticmethod
    def create(decoder_config):
        """创建解码器实例。

        Args:
            decoder_config (dict): 解码器配置，包含 type 字段。

        Returns:
            Decoder: 解码器实例。

        Raises:
            ValueError: 当解码器类型不支持时抛出。
        """
        d_type = decoder_config.get("type", "bytelevel")
        if d_type == "bytelevel":
            return ByteLevelDecoder()
        else:
            raise ValueError(f"不支持的解码器类型: {d_type}")