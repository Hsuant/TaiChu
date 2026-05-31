#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer 综合评估工具。

本脚本参考以下学术工作，对分词器进行多维评估：
- "Tokenization of Text: A Review, Evaluation, and Comparison" (Mielke et al., 2021)
- "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Tokenizers" (Rust et al., 2021)
- "A Systematic Analysis of Vocabulary and BPE Settings for Optimal Language Modeling" (Gowda & May, 2020)

评估指标包括：
- 繁殖度（Fertility）：每个词产生的 token 数量。
- 平等性（Parity）：不同文字系统下每个 token 对应的字符数。
- 重建保真度：完全匹配率和字符错误率（CER）。
- 未登录词率（OOV rate）。
- Token 分布熵与基尼系数。
- 编码速度。

数据输入支持单个 JSONL 文件或包含多个 JSONL 文件的目录。
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import jsonlines
import numpy as np
from tqdm import tqdm

# 可选依赖：editdistance 用于快速编辑距离计算。
try:
    import editdistance
    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False
    print("警告：未安装 'editdistance'，将使用 difflib 近似计算字符错误率。安装命令：pip install editdistance")

# 可选依赖：jieba 用于中文分词，以准确统计词数。
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("警告：未安装 'jieba'，中文词数将基于字符估算。安装命令：pip install jieba")

from transformers import AutoTokenizer


class TokenizerEvaluator:
    """对指定分词器在 JSONL 数据集上进行全面评估。

    属性:
        tokenizer_name: 分词器名称或路径。
        vocab_size: 分词器词汇表大小。
        text_field: JSONL 中用于提取文本的字段名。
        max_samples: 最大评估样本数，None 表示使用全部数据。
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        text_field: str = "text",
        max_samples: Optional[int] = None,
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        """初始化评估器。

        参数:
            tokenizer_name_or_path: HuggingFace 模型名或本地分词器文件路径。
            text_field: JSONL 对象中存储文本的字段名。
            max_samples: 限制加载的样本数量，用于快速测试。
            use_fast: 是否优先使用 Fast 分词器。
            trust_remote_code: 是否信任远程仓库中的自定义代码。
        """
        self.text_field = text_field
        self.max_samples = max_samples

        # 尝试使用 AutoTokenizer 加载，失败时回退到 tokenizers.Tokenizer。
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )
            self._use_raw_tokenizer = False
        except Exception as e:
            print(f"AutoTokenizer 加载失败: {e}")
            print("尝试从本地 tokenizer.json 加载...")
            from tokenizers import Tokenizer as FastTokenizer
            self.tokenizer = FastTokenizer.from_file(tokenizer_name_or_path)
            self._use_raw_tokenizer = True

        self.tokenizer_name = tokenizer_name_or_path
        self.vocab_size = self._get_vocab_size()
        self.stats: Dict[str, Any] = {}

    def _get_vocab_size(self) -> int:
        """获取词汇表大小，兼容不同类型的 tokenizer 对象。"""
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        if hasattr(self.tokenizer, "get_vocab_size"):
            return self.tokenizer.get_vocab_size()
        return len(self.tokenizer.get_vocab())

    def _encode(self, text: str) -> List[int]:
        """将文本编码为 token ID 列表，不添加特殊标记。"""
        if self._use_raw_tokenizer:
            return self.tokenizer.encode(text).ids
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, ids: List[int]) -> str:
        """将 token ID 列表解码为文本，跳过特殊标记。"""
        if self._use_raw_tokenizer:
            return self.tokenizer.decode(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _count_words(self, text: str) -> int:
        """估算文本中的词数，支持中英文混合。

        若 jieba 可用则使用其分词结果，否则将拉丁字母序列视为一个词，
        每个 CJK 字符视为一个词。
        """
        if JIEBA_AVAILABLE:
            words = jieba.lcut(text)
            # 过滤掉空白字符，只保留有意义的词。
            return len([w for w in words if w.strip()])
        # 回退方案：统计拉丁单词和单个汉字。
        latin_words = len(re.findall(r'[A-Za-z0-9]+', text))
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))
        return latin_words + cjk_chars

    def _count_chars_by_script(self, text: str) -> Dict[str, int]:
        """按文字系统粗略统计字符数量，用于平等性分析。

        返回包含 'latin', 'cjk', 'digits', 'other' 计数的字典。
        """
        latin = len(re.findall(r'[A-Za-z]', text))
        cjk = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))
        digits = len(re.findall(r'\d', text))
        other = len(text) - latin - cjk - digits
        return {"latin": latin, "cjk": cjk, "digits": digits, "other": other}

    def load_data(self, data_path: str) -> List[str]:
        """从 JSONL 文件或目录加载文本数据。

        参数:
            data_path: JSONL 文件路径或包含 .jsonl 文件的目录路径。

        返回:
            提取的文本字符串列表。

        异常:
            ValueError: 如果路径无效。
        """
        texts = []
        if os.path.isfile(data_path):
            paths = [data_path]
        elif os.path.isdir(data_path):
            paths = [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith('.jsonl')
            ]
        else:
            raise ValueError(f"无效的数据路径: {data_path}")

        print(f"正在从 {len(paths)} 个文件中加载数据...")
        for filepath in paths:
            with jsonlines.open(filepath, 'r') as reader:
                for obj in reader:
                    if self.text_field in obj:
                        texts.append(obj[self.text_field])
                        if self.max_samples and len(texts) >= self.max_samples:
                            break
            if self.max_samples and len(texts) >= self.max_samples:
                break

        print(f"已加载 {len(texts)} 条文本。")
        return texts

    def evaluate(self, texts: List[str]) -> Dict[str, Any]:
        """执行核心评估流程，收集原始统计数据。

        参数:
            texts: 待评估的文本列表。

        返回:
            包含字符数、token 数、词数、UNK 数、重建编辑距离等信息的字典。
        """
        stats: Dict[str, Any] = {
            "num_texts": len(texts),
            "total_chars": 0,
            "total_tokens": 0,
            "total_words": 0,
            "total_unk": 0,
            "token_counter": Counter(),
            "reconstruction_exact": 0,
            "reconstruction_edits": 0,
            "reconstruction_total_chars": 0,
            "script_counts": {"latin": 0, "cjk": 0, "digits": 0, "other": 0},
        }

        start_time = time.time()

        for text in tqdm(texts, desc="评估中", unit="text"):
            chars = len(text)
            words = self._count_words(text)
            stats["total_chars"] += chars
            stats["total_words"] += words

            # 按文字系统统计字符。
            script_counts = self._count_chars_by_script(text)
            for key in stats["script_counts"]:
                stats["script_counts"][key] += script_counts[key]

            # 分词并记录 token 频率。
            ids = self._encode(text)
            tokens = len(ids)
            stats["total_tokens"] += tokens
            stats["token_counter"].update(ids)

            # 检测未登录词（UNK）。
            unk_id = None
            if hasattr(self.tokenizer, "unk_token_id"):
                unk_id = self.tokenizer.unk_token_id
            elif hasattr(self.tokenizer, "token_to_id"):
                unk_id = self.tokenizer.token_to_id("<unk>")
            if unk_id is not None:
                stats["total_unk"] += ids.count(unk_id)

            # 重建保真度检测。
            decoded = self._decode(ids)
            if decoded == text:
                stats["reconstruction_exact"] += 1

            # 计算编辑距离（字符级）。
            if EDITDISTANCE_AVAILABLE:
                dist = editdistance.eval(text, decoded)
            else:
                import difflib
                matcher = difflib.SequenceMatcher(None, text, decoded)
                dist = int(
                    (1.0 - matcher.ratio()) * max(len(text), len(decoded))
                )
            stats["reconstruction_edits"] += dist
            stats["reconstruction_total_chars"] += max(len(text), len(decoded))

        elapsed = time.time() - start_time
        stats["eval_time"] = elapsed
        return stats

    def compute_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """根据原始统计数据计算各项评估指标。

        参数:
            stats: evaluate() 方法返回的原始统计字典。

        返回:
            包含繁殖度、平等性、OOV 率、分布熵、基尼系数等指标的字典。
        """
        metrics: Dict[str, Any] = {}
        n = stats["num_texts"]
        if n == 0:
            return metrics

        # 繁殖度：每个词对应的 token 数。
        metrics["fertility"] = (
            stats["total_tokens"] / stats["total_words"]
            if stats["total_words"]
            else 0.0
        )

        # 整体字符/token 比，以及按文字系统的近似值。
        metrics["char_per_token"] = (
            stats["total_chars"] / stats["total_tokens"]
            if stats["total_tokens"]
            else 0.0
        )
        for script, count in stats["script_counts"].items():
            if count > 0:
                script_tokens_share = (
                    count / stats["total_chars"] * stats["total_tokens"]
                )
                metrics[f"char_per_token_{script}"] = (
                    count / script_tokens_share if script_tokens_share else 0.0
                )

        # 未登录词率。
        metrics["unk_rate"] = (
            stats["total_unk"] / stats["total_tokens"]
            if stats["total_tokens"]
            else 0.0
        )

        # 重建保真度：完全匹配率和字符错误率。
        metrics["reconstruction_exact_match"] = stats["reconstruction_exact"] / n
        metrics["reconstruction_cer"] = (
            stats["reconstruction_edits"] / stats["reconstruction_total_chars"]
            if stats["reconstruction_total_chars"]
            else 0.0
        )

        # Token 分布熵与基尼系数。
        if stats["token_counter"]:
            total_tokens = sum(stats["token_counter"].values())
            probs = np.array(list(stats["token_counter"].values())) / total_tokens
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            max_entropy = np.log(len(stats["token_counter"]))
            metrics["token_entropy"] = entropy
            metrics["token_entropy_norm"] = (
                entropy / max_entropy if max_entropy > 0 else 0.0
            )
            # 计算基尼系数（0 表示完全均匀分布，1 表示极度集中）。
            sorted_probs = np.sort(probs)
            cum_probs = np.cumsum(sorted_probs)
            m = len(probs)
            indices = np.arange(1, m + 1)
            gini = (2 * np.sum(indices * sorted_probs) - (m + 1) * np.sum(sorted_probs)) / (
                m * np.sum(sorted_probs)
            )
            metrics["token_gini"] = gini
        else:
            metrics["token_entropy"] = 0.0
            metrics["token_entropy_norm"] = 0.0
            metrics["token_gini"] = 0.0

        # 词汇表使用情况。
        metrics["vocab_size_actual"] = len(stats["token_counter"])
        metrics["vocab_size_total"] = self.vocab_size
        metrics["vocab_utilization"] = (
            metrics["vocab_size_actual"] / self.vocab_size
            if self.vocab_size
            else 0.0
        )

        # 编码速度（tokens/秒）。
        metrics["encoding_speed_tokens_per_sec"] = (
            stats["total_tokens"] / stats["eval_time"]
            if stats["eval_time"]
            else 0.0
        )

        # 人类可读的压缩比。
        metrics["compression_ratio"] = (
            stats["total_chars"] / stats["total_tokens"]
            if stats["total_tokens"]
            else 0.0
        )

        return metrics

    def test_chat_template(self, num_samples: int = 3) -> Dict[str, Any]:
        """测试 tokenizer 的对话模板（chat_template）功能。

        该测试仅当 tokenizer 具有 chat_template 属性时执行。
        使用示例多轮对话来应用模板，验证生成文本是否包含预期的特殊标记，
        并确保编码-解码后仍可保留模板结构。

        参数:
            num_samples: 生成的测试对话样本数量（默认 3）。

        返回:
            包含测试结果的字典，键为测试名称，值为详情或状态。
        """
        results: Dict[str, Any] = {"enabled": False, "template_available": False, "samples": []}

        # 检查 tokenizer 是否包含 chat_template 属性
        if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
            results["error"] = "该 tokenizer 未设置 chat_template"
            return results

        results["template_available"] = True
        results["enabled"] = True

        # 保存原始设置，测试后恢复
        original_skip_special = getattr(self.tokenizer, "skip_special_tokens", True)
        self.tokenizer.skip_special_tokens = False  # 确保解码时保留特殊 token

        try:
            for i in range(num_samples):
                # 构造一个简单的多轮对话样本，包含系统消息、用户消息和助手回复
                messages = [
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": "你好，请介绍一下你自己。"},
                    {"role": "assistant", "content": "我是人工智能助手，可以回答问题。"},
                    {"role": "user", "content": "今天的天气怎么样？"},
                ]
                try:
                    # 1. 应用模板生成格式化文本
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    # 2. 第一次编码
                    ids_1 = self.tokenizer.encode(formatted, add_special_tokens=False)
                    # 3. 解码
                    decoded = self.tokenizer.decode(ids_1, skip_special_tokens=False)
                    # 4. 第二次编码（对解码后的文本再次编码）
                    ids_2 = self.tokenizer.encode(decoded, add_special_tokens=False)
                    # 5. 检查 Roundtrip 一致性
                    roundtrip_ok = (ids_1 == ids_2)
                    results["samples"].append({
                        "messages": messages,
                        "formatted_text": formatted,
                        "decoded_text": decoded,
                        "encoded_length": len(ids_1),
                        "roundtrip_ok": roundtrip_ok,
                        "ids_1": ids_1,
                        "ids_2": ids_2,
                    })
                except Exception as e:
                    results["samples"].append({"messages": messages, "error": str(e)})
        finally:
            # 恢复原始设置
            self.tokenizer.skip_special_tokens = original_skip_special

        return results

    def test_special_tokens(self) -> Dict[str, Any]:
        """测试 tokenizer 的特殊 token（如 BOS、EOS、UNK、PAD 等）功能。

        验证特殊 token 的 ID 与文本是否对应，编码解码后能否正确还原，
        并检查特殊 token 是否在词汇表中存在。

        返回:
            包含各项测试结果的字典，结构为：
            {
                "bos": {"id": ..., "text": ..., "encode_ok": ..., "decode_ok": ...},
                "eos": {...},
                "unk": {...},
                "pad": {...},
                "additional_special_tokens": [...]
            }
        """
        results: Dict[str, Any] = {}

        # 定义要测试的特殊 token 属性及其期望的常见名称
        special_attrs = {
            "bos": "bos_token",
            "eos": "eos_token",
            "unk": "unk_token",
            "pad": "pad_token",
        }

        for token_type, attr_name in special_attrs.items():
            token_info = {"id": None, "text": None, "exists_in_vocab": False,
                          "encode_ok": False, "decode_ok": False, "error": None}
            try:
                token_text = getattr(self.tokenizer, attr_name, None)
                token_id = getattr(self.tokenizer, f"{token_type}_token_id", None)

                token_info["text"] = token_text
                token_info["id"] = token_id

                if token_text is not None and token_id is not None:
                    # 验证词汇表中 ID 与文本的映射
                    try:
                        decoded_text = self.tokenizer.decode([token_id])
                        token_info["decode_ok"] = (decoded_text == token_text)
                    except Exception:
                        token_info["decode_ok"] = False

                    # 编码文本应得到对应的 ID
                    try:
                        encoded_id = self.tokenizer.encode(token_text, add_special_tokens=False)
                        token_info["encode_ok"] = (len(encoded_id) == 1 and encoded_id[0] == token_id)
                    except Exception:
                        token_info["encode_ok"] = False

                    # 检查 ID 是否在词汇表内
                    token_info["exists_in_vocab"] = token_id < self.vocab_size
            except Exception as e:
                token_info["error"] = str(e)

            results[token_type] = token_info

        # 测试附加特殊 token（如 <|im_start|> 等）
        add_special = getattr(self.tokenizer, "additional_special_tokens", None)
        if add_special:
            results["additional_special_tokens"] = []
            for i, token_text in enumerate(add_special):
                info = {"text": token_text, "id": None, "encode_ok": False, "decode_ok": False}
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids(token_text)
                    info["id"] = token_id
                    if token_id != self.tokenizer.unk_token_id:
                        decoded = self.tokenizer.decode([token_id])
                        info["decode_ok"] = (decoded == token_text)
                        encoded = self.tokenizer.encode(token_text, add_special_tokens=False)
                        info["encode_ok"] = (len(encoded) == 1 and encoded[0] == token_id)
                except Exception as e:
                    info["error"] = str(e)
                results["additional_special_tokens"].append(info)

        return results

    def run_speed_benchmark(self, texts: List[str], num_iterations: int = 5,
                            warmup: int = 2) -> float:
        """运行独立的编码速度基准测试。

        参数:
            texts: 文本列表，仅使用前 1000 条作为测试样本。
            num_iterations: 正式测试的迭代次数。
            warmup: 预热迭代次数，不计入最终速度。

        返回:
            平均每秒编码的 token 数。
        """
        sample_texts = texts[: min(1000, len(texts))]
        if not sample_texts:
            return 0.0

        # 预热运行，消除首次调用的额外开销。
        for _ in range(warmup):
            for text in sample_texts:
                self._encode(text)

        # 正式计时。
        start = time.perf_counter()
        for _ in range(num_iterations):
            for text in sample_texts:
                self._encode(text)
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(self._encode(t)) for t in sample_texts) * num_iterations
        return total_tokens / elapsed if elapsed > 0 else 0.0

    def print_report(
            self,
            metrics: Dict[str, Any],
            speed: float,
            chat_template_results: Optional[Dict[str, Any]] = None,
            special_token_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """打印格式化的评估报告，包含核心指标、对话模板测试与特殊标签测试摘要。

        所有输出严格对齐，标签宽度 30，数值宽度 15。
        超过宽度的值会被截断并在末尾添加省略号。
        """
        label_width = 30
        value_width = 15

        def truncate(s: str, width: int) -> str:
            """截断字符串至指定宽度，超长部分替换为'…'。"""
            if len(s) <= width:
                return s
            return s[:width - 1] + "…"

        def fmt(label: str, value_str: str) -> str:
            """生成对齐的“标签 : 数值”行，自动截断过长的值。"""
            return f"{label:<{label_width}} : {truncate(value_str, value_width):>{value_width}}"

        # ---- 核心指标报告 ----
        print("\n" + "=" * 70)
        print(f"分词器评估报告: {self.tokenizer_name}")
        print("=" * 70)

        print(fmt("文本数量", f"{metrics.get('num_texts', 0)}"))
        print(fmt("总字符数", f"{metrics.get('total_chars', 0):,}"))
        print(fmt("总 token 数", f"{metrics.get('total_tokens', 0):,}"))
        print(fmt("总词数（估算）", f"{metrics.get('total_words', 0):,}"))
        print(fmt("词汇表大小", f"{self.vocab_size:,}"))
        print(fmt("实际使用词汇数", f"{metrics.get('vocab_size_actual', 0):,}"))
        print(fmt("词汇利用率", f"{metrics.get('vocab_utilization', 0):.2%}"))
        print(fmt("繁殖度 (Fertility)", f"{metrics.get('fertility', 0):.4f}"))
        print(fmt("字符/Token", f"{metrics.get('char_per_token', 0):.4f}"))
        print(fmt("字符/Token (拉丁)", f"{metrics.get('char_per_token_latin', 0):.4f}"))
        print(fmt("字符/Token (CJK)", f"{metrics.get('char_per_token_cjk', 0):.4f}"))
        print(fmt("OOV/UNK 率", f"{metrics.get('unk_rate', 0):.4%}"))
        print(fmt("重建完全匹配率", f"{metrics.get('reconstruction_exact_match', 0):.4%}"))
        print(fmt("重建 CER", f"{metrics.get('reconstruction_cer', 0):.6f}"))
        print(fmt("Token 熵 (nats)", f"{metrics.get('token_entropy', 0):.4f}"))
        print(fmt("归一化熵", f"{metrics.get('token_entropy_norm', 0):.4f}"))
        print(fmt("Token 基尼系数", f"{metrics.get('token_gini', 0):.4f}"))
        print(fmt("编码速度", f"{speed:,.0f} tokens/秒"))
        print("=" * 70)

        # ---- 对话模板测试摘要 ----
        print("\n" + "-" * 70)
        print("对话模板测试摘要")
        print("-" * 70)

        if chat_template_results is None:
            chat_template_results = {}

        if chat_template_results.get("enabled"):
            print(fmt("模板可用", "是"))
            samples = chat_template_results.get("samples", [])
            print(fmt("测试样本数", str(len(samples))))
            for idx, sample in enumerate(samples):
                status = "通过" if sample.get("roundtrip_ok") else "失败"
                label = f"样本 {idx + 1}"
                print(fmt(label, status))
        else:
            error_msg = chat_template_results.get("error", "未设置")
            # 截断错误消息以保持对齐
            print(fmt("模板可用", f"否 ({error_msg})"))

        # ---- 特殊标签测试摘要 ----
        print("\n" + "-" * 70)
        print("特殊标签测试摘要")
        print("-" * 70)

        if special_token_results is None:
            special_token_results = {}

        # 标准特殊标记（BOS, EOS, UNK, PAD）
        for token_type in ["bos", "eos", "unk", "pad"]:
            info = special_token_results.get(token_type, {})
            status = "通过" if info.get("encode_ok") and info.get("decode_ok") else "失败"
            token_id = info.get("id")
            token_text = info.get("text", "?")
            # 构建简短的值字符串：状态 + ID
            value = f"{status} (ID:{token_id})" if token_id is not None else status
            print(fmt(token_type.upper(), value))

        # 附加特殊标记（如 <|im_start|> 等）
        additional_tokens = special_token_results.get("additional_special_tokens", [])
        for token in additional_tokens:
            token_text = token.get("text", "?")
            status = "通过" if token.get("encode_ok") and token.get("decode_ok") else "失败"
            token_id = token.get("id")
            value = f"{status} (ID:{token_id})" if token_id is not None else status
            # 标签限制30字符，对过长的 token 文本进行截断
            label = f"附加 token {token_text}"
            if len(label) > label_width:
                label = label[:label_width - 1] + "…"
            print(fmt(label, value))

        print("-" * 70)

    def to_json(self, metrics: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """将指标保存为 JSON 文件或输出到标准输出。

        参数:
            metrics: 指标字典。
            output_path: 保存路径，若为 None 则打印到终端。
        """
        out = {
            k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v))
            for k, v in metrics.items()
        }
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"指标已保存至 {output_path}")
        else:
            print(json.dumps(out, indent=2, ensure_ascii=False))

    def evaluate_from_path(self, data_path: str, run_speed: bool = True,
                           output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """完整的评估流水线：加载数据、评估、保存 JSON（若指定）、输出报告。

        参数:
            data_path: JSONL 数据路径。
            run_speed: 是否执行速度基准测试。
            output_json_path: 保存所有指标的 JSON 文件路径（可选）。

        返回:
            合并原始统计与计算指标的完整字典。
        """
        texts = self.load_data(data_path)
        if not texts:
            print("未加载到任何文本，退出。")
            return {}

        raw_stats = self.evaluate(texts)
        metrics = self.compute_metrics(raw_stats)
        # 将部分原始统计量加入指标字典，便于报告展示。
        metrics.update({
            "num_texts": raw_stats["num_texts"],
            "total_chars": raw_stats["total_chars"],
            "total_tokens": raw_stats["total_tokens"],
            "total_words": raw_stats["total_words"],
        })

        speed = 0.0
        if run_speed:
            print("正在运行速度基准测试...")
            speed = self.run_speed_benchmark(texts)

        print("正在进行对话模板测试……")
        chat_template_results = self.test_chat_template()

        print("正在进行特殊标签测试……")
        special_token_results = self.test_special_tokens()

        # 合并速度指标
        full_metrics = {
            **metrics,
            "encoding_speed": speed,
            "chat_template_test": chat_template_results,
            "special_tokens_test": special_token_results,
        }

        # 先保存 JSON（若指定路径），再打印报告。
        if output_json_path:
            self.to_json(full_metrics, output_json_path)

        self.print_report(
            metrics=metrics,
            speed=speed,
            chat_template_results=chat_template_results,
            special_token_results=special_token_results,
        )
        return full_metrics


def main() -> None:
    """命令行入口函数。"""
    parser = argparse.ArgumentParser(
        description="分词器综合评估工具"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL 文件或包含 .jsonl 文件的目录路径。",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace 分词器名称或本地 tokenizer.json 路径。",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="JSON 对象中存储文本的字段名（默认: text）。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="限制评估的样本数量，用于快速测试。",
    )
    parser.add_argument(
        "--no_speed",
        action="store_true",
        help="跳过速度基准测试。",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="将指标保存为 JSON 文件的路径。",
    )
    parser.add_argument(
        "--use_slow",
        action="store_true",
        help="使用慢速分词器而非 Fast 分词器。",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="加载分词器时信任远程仓库中的自定义代码。",
    )

    args = parser.parse_args()

    evaluator = TokenizerEvaluator(
        tokenizer_name_or_path=args.tokenizer,
        text_field=args.text_field,
        max_samples=args.max_samples,
        use_fast=not args.use_slow,
        trust_remote_code=args.trust_remote_code,
    )

    # 执行评估：内部会先保存 JSON（若指定），再打印报告。
    evaluator.evaluate_from_path(
        data_path=args.data_path,
        run_speed=not args.no_speed,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()