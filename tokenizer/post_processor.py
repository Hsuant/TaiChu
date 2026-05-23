"""
后处理器工厂模块。
根据配置文件动态构建对话模板处理器，支持自定义角色与特殊Token，
适用于仅含对话与思考（无工具调用）的工业级场景。
"""

from typing import Dict, Any, Optional
from tokenizers.processors import TemplateProcessing


class PostProcessorFactory:
    """基于配置的对话模板处理器工厂。

    支持通过 YAML 配置自定义 single/pair 模板字符串，
    模板格式遵循 tokenizers.TemplateProcessing 规范：
        - $A 表示第一个输入序列
        - $B 表示第二个输入序列（仅 pair 模板）
    示例：
        single_template: "<|im_start|>assistant\n$A<|im_end|>"
        pair_template: "<|im_start|>user\n$A<|im_end|>\n<|im_start|>assistant\n$B<|im_end|>"
    """

    @staticmethod
    def create(
        config: Dict[str, Any],
        tokenizer: "Tokenizer"  # type: ignore  # noqa: F821
    ) -> Optional[TemplateProcessing]:
        """根据配置创建后处理器。

        Args:
            config: YAML 中 post_processor 的配置字典，可能包含：
                - single_template: 单序列模板（可选）
                - pair_template: 双序列模板（可选）
                - special_tokens: 模板中出现的特殊 Token 列表（可选）
            tokenizer: 已训练的 tokenizer 实例，用于获取特殊 Token 的 ID。

        Returns:
            配置好的 TemplateProcessing 实例，若配置为空则返回 None。

        Raises:
            ValueError: 当模板中的特殊 Token 未在 tokenizer 中找到时抛出。
        """
        # 检查是否有模板定义
        single_template = config.get("single_template", None)
        pair_template = config.get("pair_template", None)

        if single_template is None and pair_template is None:
            # 未配置模板，不使用后处理器
            return None

        # 获取模板中需要映射为 ID 的特殊 Token
        specified_special_tokens = config.get("special_tokens", [])
        if not specified_special_tokens:
            # 若未显式指定，尝试从模板中自动提取（格式为 <...> 的 Token）
            # 但为保持工业级的明确性，建议用户在配置中显式列出
            print("警告：未在 post_processor 配置中指定 special_tokens，"
                  "可能无法正确映射模板中的特殊 Token。")

        # 构建 (token, token_id) 列表，供 TemplateProcessing 使用
        special_pairs = []
        for token in specified_special_tokens:
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"特殊 Token '{token}' 在词表中未找到，"
                                 "请确保它在 trainer.special_tokens 列表中。")
            special_pairs.append((token, token_id))

        # 创建 TemplateProcessing 实例
        processor = TemplateProcessing(
            single=single_template or "",
            pair=pair_template or "",
            special_tokens=special_pairs,
        )
        return processor