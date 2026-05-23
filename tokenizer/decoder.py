"""
解码器工厂模块。根据配置创建解码器，用于将Token ID序列转换回原始文本。
"""
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class DecoderFactory:
    """解码器工厂，根据类型返回对应的解码器实例。"""

    @staticmethod
    def create(decoder_config):
        """
        创建解码器实例。

        Args:
            decoder_config (dict): 解码器配置。

        Returns:
            Decoder: 解码器实例。

        Raises:
            ValueError: 当遇到不支持的解码器类型时抛出。
        """
        d_type = decoder_config.get("type", "bytelevel")
        if d_type == "bytelevel":
            return ByteLevelDecoder()
        else:
            raise ValueError(f"不支持的解码器类型: {d_type}")