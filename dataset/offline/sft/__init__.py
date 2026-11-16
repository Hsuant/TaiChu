"""通用 SFT 离线数据准备包。

公开 API 只包含配置对象、配置加载器和完整构建流程；不暴露任何平台专用
适配器，确保 SFT 数据处理在 dataset 层保持通用。
"""

from .config import RawSourceConfig, SFTBuildConfig, load_sft_build_config
from .pipeline import SFTDataPreparation

__all__ = [
    "SFTBuildConfig",
    "SFTDataPreparation",
    "RawSourceConfig",
    "load_sft_build_config",
]
