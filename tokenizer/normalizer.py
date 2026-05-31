"""
规范化器工厂模块。

根据配置动态构建 Unicode 规范化器，支持 NFKC、正则替换、去首尾空格等步骤，
并可组合为序列化流水线。
"""

from tokenizers import Regex
from tokenizers.normalizers import NFKC, Replace, Strip, Sequence


class NormalizerFactory:
    """规范化器工厂，根据 YAML 配置返回对应的 Normalizer 实例。

    配置格式示例：
        normalizer:
          type: "sequence"
          steps:
            - type: "nfkc"
            - type: "replace"
              pattern: "\\s+"
              content: " "
            - type: "strip"
    """

    @staticmethod
    def create(norm_config):
        """创建规范化器实例。

        Args:
            norm_config (dict): 规范化器配置字典，包含 type 及相关参数。
                当 type 为 "none" 时返回 None（不做规范化）。

        Returns:
            normalizers.Normalizer 或其子类，若配置为 none 则返回 None。

        Raises:
            ValueError: 遇到不支持的类型时抛出。
        """
        n_type = norm_config.get("type", "nfkc")

        if n_type == "none":
            return None

        if n_type == "nfkc":
            return NFKC()

        if n_type == "sequence":
            steps = []
            for step in norm_config.get("steps", []):
                s_type = step["type"]
                if s_type == "nfkc":
                    steps.append(NFKC())
                elif s_type == "replace":
                    pattern = Regex(step["pattern"])
                    content = step.get("content", "")
                    steps.append(Replace(pattern, content))
                elif s_type == "strip":
                    steps.append(Strip())
                else:
                    raise ValueError(f"不支持的规范化步骤类型: {s_type}")
            return Sequence(steps) if steps else None

        raise ValueError(f"不支持的规范化器类型: {n_type}")