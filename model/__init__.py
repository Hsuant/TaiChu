"""XuanYan 模型包。

包含构建 GPT 风格语言模型所需的所有模块：
- TokenEmbedding: 词嵌入
- RoPEPositionEncoding: 旋转位置编码
- RMSNorm: RMS归一化
- TaiChuCausalSelfAttention: 因果自注意力
- SwiGLUFeedForward: SwiGLU前馈网络
- TransformerBlock: 单个解码器层
- TaiChuOutputHead: 输出投影头
- TaiChuModel: 完整的语言模型
"""

from model.embedding import TokenEmbedding
from model.positional_encoding import RoPEPositionEncoding
from model.normalization import RMSNorm
from model.attention import TaiChuCausalSelfAttention
from model.feedforward import SwiGLUFeedForward
from model.transformer_block import TaiChuBlock
from model.output_head import TaiChuOutputHead
from model.model import TaiChuModel

__all__ = [
    "TokenEmbedding",
    "RoPEPositionEncoding",
    "RMSNorm",
    "TaiChuCausalSelfAttention",
    "SwiGLUFeedForward",
    "TaiChuBlock",
    "TaiChuOutputHead",
    "TaiChuModel",
]