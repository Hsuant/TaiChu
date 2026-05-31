# -*- coding: utf-8 -*-
"""预分词器工厂模块。

根据配置创建不同类型的预分词器。支持 ByteLevel 以及 Llama 3 风格的
数字拆分（R2L Digit Splitting），以提升数学推理和代码能力。
"""

from tokenizers.pre_tokenizers import ByteLevel, Whitespace, Split, Sequence
from tokenizers import Regex, SplitDelimiterBehavior


class PreTokenizerFactory:
    """预分词器工厂，根据类型返回对应的预分词器实例。

    新增特性：
        - digit_split: 启用 Llama 3 风格的数字拆分（R2L）
        - max_digit_group: 数字分组大小（3 表示千分位风格）
    """

    @staticmethod
    def create(pre_tokenizer_config):
        """创建预分词器实例。

        Args:
            pre_tokenizer_config (dict): 预分词器配置，包含 type 及相关参数。

        Returns:
            PreTokenizer: 预分词器实例。
        """
        p_type = pre_tokenizer_config.get("type", "bytelevel")

        if p_type == "bytelevel":
            # 获取数字拆分配置
            digit_split = pre_tokenizer_config.get("digit_split", False)
            max_digit_group = pre_tokenizer_config.get("max_digit_group", 3)

            # 构建预分词步骤序列
            steps = []

            # 步骤 1：数字拆分（可选）
            if digit_split:
                # Llama 3 风格的 R2L 数字拆分正则，增加负向后顾提高鲁棒性
                # 匹配 1-3 位数字，从右向左分组（R2L: Right-to-Left）
                # 这确保数字进位边界正确，对于算术推理至关重要
                # 确保数字前面不是点、斜线、冒号、连字符、字母，
                # 以避免拆分 IP 地址 (192.168.1.1)、版本号 (v1.2.3)、日期 (2025-01-01) 等。
                # 负向后顾 (?<![.\/\-:a-zA-Z]) 表示前面不能是 . / - : 字母
                # 注意：仍可能拆分某些边界情况，但已显著降低误拆分率。
                # 参考：Llama 3 tokenizer 的 r"\d{1,3}(?=(\d{3})*\b)"
                digit_pattern = rf"(?<![.\/\-:a-zA-Z])\d{{1,{max_digit_group}}}(?=(\d{{{max_digit_group}}})*\b)"
                steps.append(
                    Split(
                        pattern=Regex(digit_pattern),
                        behavior=SplitDelimiterBehavior.ISOLATED,
                        invert=False
                    )
                )
                print(f"已启用数字拆分（R2L），最大分组: {max_digit_group} 位")

            # 步骤 2：ByteLevel 预分词
            steps.append(
                ByteLevel(
                    add_prefix_space=pre_tokenizer_config.get("add_prefix_space", False),
                    use_regex=pre_tokenizer_config.get("use_regex", True)
                )
            )

            # 若只有一个步骤，直接返回该步骤
            if len(steps) == 1:
                return steps[0]

            return Sequence(steps)

        elif p_type == "whitespace":
            return Whitespace()
        else:
            raise ValueError(f"不支持的预分词器类型: {p_type}")