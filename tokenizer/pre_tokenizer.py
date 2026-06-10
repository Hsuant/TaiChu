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
            punctuation_split = pre_tokenizer_config.get("punctuation_split", False)

            # 构建预分词步骤序列
            steps = []

            # 步骤 1：数字拆分（可选）
            if digit_split:
                # 安全前缀：防止拆分 IP 地址、日期、版本号等含分隔符的数字串
                safe_prefix = r"(?<![.\/\-:a-zA-Z])"
                # 零宽位置拆分正则：
                #   (?<=\d)          当前位置左边必须是数字
                #   (?=(\d{3})+(?!\d)) 当前位置右边必须是3的倍数个连续数字，且之后不是数字
                # 这样匹配的是“内部插入点”，通过 REMOVED 行为直接分割，
                # 不会引入额外 token，实现了 R2L 的千分位分组。
                digit_pattern = safe_prefix + r"(?<=\d)(?=(\d{3})+(?!\d))"
                steps.append(
                    Split(
                        pattern=Regex(digit_pattern),
                        behavior=SplitDelimiterBehavior.REMOVED,
                        invert=False
                    )
                )

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