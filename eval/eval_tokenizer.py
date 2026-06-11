#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM Tokenizer 专业评估工具（带进度条与日志集成）。

本脚本用于全面评估分词器（Tokenizer）在大型语言模型（LLM）中的性能。
它会测试压缩效率、覆盖率、重建保真度、分布统计、特殊 Token 和聊天模板功能，
并将结果输出为结构化的 JSON 报告。

依赖：
    - transformers
    - Levenshtein (可选)
    - jieba (可选)
    - tqdm

使用方法：
    python -m eval.eval_tokenizer --tokenizer_path /path/to/tokenizer --test_file /path/to/texts.txt --output report.json
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# 进度条
from tqdm import tqdm

# ----------------------------------------------------------------------
# 可选依赖
# ----------------------------------------------------------------------
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

try:
    import Levenshtein
    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False

# ----------------------------------------------------------------------
# 日志管理器（集成 LogManager）
# ----------------------------------------------------------------------
try:
    from utils.logger import LogManager
    _HAS_LOGMANAGER = True
except ImportError:
    _HAS_LOGMANAGER = False
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LogManager = None


# ----------------------------------------------------------------------
# 内置编辑距离（Levenshtein）回退实现
# ----------------------------------------------------------------------
def _edit_distance_builtin(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance_builtin(s2, s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        current_row = [i]
        for j, c2 in enumerate(s2, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def edit_distance(s1: str, s2: str) -> int:
    if _HAS_LEVENSHTEIN:
        return Levenshtein.distance(s1, s2)
    return _edit_distance_builtin(s1, s2)


# ----------------------------------------------------------------------
# 字符分类正则表达式
# ----------------------------------------------------------------------
_RE_LATIN = re.compile(r"[\u0041-\u005A\u0061-\u007A\u00C0-\u024F\u1E00-\u1EFF]")
_RE_FULLWIDTH_LATIN = re.compile(r"[\uFF21-\uFF3A\uFF41-\uFF5A]")
_RE_FULLWIDTH_DIGIT = re.compile(r"[\uFF10-\uFF19]")
_RE_DIGITS = re.compile(r"\d")
_RE_CJK = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF"
    r"\u3040-\u309F\u30A0-\u30FF"
    r"\uAC00-\uD7AF"
    r"\u3000-\u303F]"
)


def classify_char(char: str) -> str:
    if _RE_FULLWIDTH_LATIN.match(char):
        return "latin"
    if _RE_FULLWIDTH_DIGIT.match(char):
        return "digits"
    if _RE_LATIN.match(char):
        return "latin"
    if _RE_CJK.match(char):
        return "cjk"
    if _RE_DIGITS.match(char):
        return "digits"
    return "other"


# ----------------------------------------------------------------------
# 默认特殊 Token 配置
# ----------------------------------------------------------------------
_DEFAULT_SPECIAL_TOKENS = {
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "unk_token": "<unk>",
    "pad_token": "<|im_end|>",
}

# ----------------------------------------------------------------------
# 聊天模板测试样本
# ----------------------------------------------------------------------
_CHAT_TEST_SAMPLES = [
    {
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请介绍一下你自己。"},
            {"role": "assistant", "content": "我是人工智能助手，可以回答问题。"},
            {"role": "user", "content": "今天的天气怎么样？"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "Tell me more about it."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "请解释一下量子计算。"},
        ]
    },
]


# ----------------------------------------------------------------------
# 核心评估类
# ----------------------------------------------------------------------
class TokenizerEvaluator:
    def __init__(self, tokenizer_path: str, test_files: List[str],
                 special_tokens: Optional[Dict[str, str]] = None,
                 log_manager: Optional[Any] = None):
        """初始化评估器。

        Args:
            tokenizer_path: 分词器路径或 HuggingFace 模型名称。
            test_files: 测试文本文件路径列表。
            special_tokens: 自定义特殊 Token 映射。
            log_manager: LogManager 实例，若为 None 则创建默认日志。
        """
        # 日志管理
        self.log = None
        self._fallback_logger = None
        if log_manager is not None:
            self.log = log_manager
        elif _HAS_LOGMANAGER and LogManager is not None:
            self.log = LogManager(log_dir="./experiments/logs", tensorboard=False, log_file="eval.log")
            self.log.info("使用 LogManager 记录日志")
        else:
            self._fallback_logger = logging.getLogger(__name__)

        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.texts = self._load_texts(test_files)      # 内部已含进度条
        self.num_texts = len(self.texts)
        self.total_chars = sum(len(text) for text in self.texts)
        self.total_tokens = 0
        self.total_words = 0
        self.token_freq = collections.Counter()
        self.unk_token_id = self._detect_unk_id()
        self.special_tokens = special_tokens if special_tokens else _DEFAULT_SPECIAL_TOKENS

        # 批量编码获取统计信息（性能优化，内部显示进度条）
        self._encode_all_batch()

    def _log(self, level: str, msg: str):
        """内部日志记录，兼容 LogManager 和回退 logger。"""
        if self.log is not None:
            getattr(self.log, level.lower())(msg)
        elif self._fallback_logger is not None:
            getattr(self._fallback_logger, level.lower())(msg)

    def _load_tokenizer(self, path: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("请安装 transformers 库：pip install transformers") from e

        if os.path.exists(path):
            self._log("info", f"从本地路径加载分词器：{path}")
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
        else:
            self._log("info", f"本地路径 {path} 不存在，尝试从 HuggingFace Hub 加载...")
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tokenizer

    def _load_texts(self, paths: List[str]) -> List[str]:
        """加载文本，使用进度条显示解析进度。"""
        all_texts = []
        # 先收集所有需要解析的文件
        file_list = []
        for path in paths:
            if not os.path.exists(path):
                self._log("warning", f"路径不存在，已跳过: {path}")
                continue
            if os.path.isdir(path):
                found = glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.jsonl"))
                if not found:
                    self._log("warning", f"目录下未找到 .json 或 .jsonl 文件: {path}")
                file_list.extend(found)
            else:
                file_list.append(path)

        self._log("info", f"共发现 {len(file_list)} 个文件待解析")
        # 使用 tqdm 遍历文件
        for file_path in tqdm(file_list, desc="加载文本文件", unit="file"):
            texts_from_file = self._parse_jsonl_or_json(file_path)
            all_texts.extend(texts_from_file)

        all_texts = [t for t in all_texts if t.strip()]
        self._log("info", f"共加载 {len(all_texts)} 条有效测试文本（来自 {len(paths)} 个路径）")
        return all_texts

    def _parse_jsonl_or_json(self, file_path: str) -> List[str]:
        texts = []
        _, ext = os.path.splitext(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if ext.lower() == ".jsonl":
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if "text" in obj and isinstance(obj["text"], str):
                                texts.append(obj["text"])
                        except json.JSONDecodeError as e:
                            self._log("warning", f"跳过 {file_path} 第 {line_num} 行 JSON 解析错误: {e}")
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, item in enumerate(data):
                            if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                                texts.append(item["text"])
                    elif isinstance(data, dict) and "text" in data:
                        if isinstance(data["text"], str):
                            texts.append(data["text"])
                    else:
                        self._log("warning", f"文件 {file_path} 不是 JSON 数组或包含 text 的对象，已忽略")
        except Exception as e:
            self._log("error", f"读取文件 {file_path} 失败: {e}")
        return texts

    def _detect_unk_id(self) -> Optional[int]:
        if hasattr(self.tokenizer, "unk_token") and self.tokenizer.unk_token is not None:
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        unk_str = self.special_tokens.get("unk_token")
        if unk_str:
            return self.tokenizer.convert_tokens_to_ids(unk_str)
        return None

    def _count_words(self, text: str) -> int:
        contains_cjk = bool(_RE_CJK.search(text))
        if contains_cjk and _HAS_JIEBA:
            words = jieba.lcut(text)
            return sum(1 for w in words if w.strip())
        elif contains_cjk:
            clean = re.sub(r'\s+', '', text)
            cjk_chars = sum(1 for ch in clean if _RE_CJK.match(ch))
            non_cjk_words = len(re.findall(r'[^\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uac00-\ud7af]+', text))
            return cjk_chars + non_cjk_words
        else:
            return len(text.split())

    def _encode_all_batch(self):
        """批量编码所有文本，显示进度条。"""
        self._log("info", "开始批量编码所有文本...")
        # 一次性编码（transformers 的 encode_batch 本质就是 tokenizer(texts)）
        encodings = self.tokenizer(self.texts, add_special_tokens=True, padding=False, truncation=False)
        all_ids = encodings.input_ids

        total_tokens = 0
        freq = collections.Counter()
        for ids in all_ids:
            total_tokens += len(ids)
            freq.update(ids)

        # 单词统计（逐条，显示进度条）
        total_words = 0
        with tqdm(total=len(self.texts), desc="统计单词数", unit="text") as pbar:
            for text in self.texts:
                total_words += self._count_words(text)
                pbar.update(1)
                # 每 10% 记录一次日志
                if pbar.n % max(1, len(self.texts)//10) == 0 and pbar.n > 0:
                    self._log("info", f"单词统计进度: {pbar.n}/{len(self.texts)} ({100*pbar.n/len(self.texts):.1f}%)")

        self.total_tokens = total_tokens
        self.total_words = total_words
        self.token_freq = freq
        self._log("info", f"批量编码完成：总 Token {total_tokens}，总单词 {total_words}")

    # ---------- 压缩 / 效率指标 ----------
    def compute_compression_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_chars / self.total_tokens

    def compute_char_per_token_by_category(self) -> Dict[str, float]:
        char_counts = {"latin": 0, "cjk": 0, "digits": 0, "other": 0}
        for text in self.texts:
            for ch in text:
                cat = classify_char(ch)
                char_counts[cat] += 1
        result = {}
        if self.total_tokens == 0:
            for k in char_counts:
                result[f"char_per_token_{k}"] = 0.0
            result["char_per_token"] = 0.0
            return result

        for cat, count in char_counts.items():
            result[f"char_per_token_{cat}"] = count / self.total_tokens
        result["char_per_token"] = self.total_chars / self.total_tokens
        return result

    def compute_fertility(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_words / self.total_tokens

    def measure_encoding_speed(self, warmup: int = 1, repeats: int = 3) -> Tuple[float, float]:
        """批量编码测速，显示进度条。"""
        texts = self.texts
        self._log("info", f"开始编码速度测试（预热 {warmup} 次，重复 {repeats} 次）")
        # 预热
        for _ in tqdm(range(warmup), desc="预热", unit="次"):
            _ = self.tokenizer(texts, padding=False, truncation=False)

        total_time = 0.0
        total_tokens = 0
        total_chars = 0
        # 实际测量
        for i in tqdm(range(repeats), desc="测速中", unit="次"):
            start = time.perf_counter()
            encodings = self.tokenizer(texts, padding=False, truncation=False)
            ids_list = encodings.input_ids
            end = time.perf_counter()
            total_time += (end - start)
            for ids, txt in zip(ids_list, texts):
                total_tokens += len(ids)
                total_chars += len(txt)

        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        avg_chars_per_sec = total_chars / total_time if total_time > 0 else 0.0
        self._log("info", f"编码速度: {avg_tokens_per_sec:.2f} tokens/秒, {avg_chars_per_sec:.2f} chars/秒")
        return avg_tokens_per_sec, avg_chars_per_sec

    # ---------- 覆盖率指标 ----------
    def compute_unk_rate(self) -> float:
        if self.total_tokens == 0 or self.unk_token_id is None:
            return 0.0
        unk_count = self.token_freq.get(self.unk_token_id, 0)
        return unk_count / self.total_tokens

    def compute_vocab_utilization(self) -> Dict[str, Any]:
        vocab_actual = len(self.token_freq)
        vocab_total = self.tokenizer.vocab_size if hasattr(self.tokenizer, "vocab_size") else None
        if vocab_total is None:
            try:
                vocab_total = len(self.tokenizer.get_vocab())
            except Exception:
                vocab_total = 0
        utilization = vocab_actual / vocab_total if vocab_total > 0 else 0.0
        return {
            "vocab_size_actual": vocab_actual,
            "vocab_size_total": vocab_total,
            "vocab_utilization": utilization,
        }

    # ---------- 重建保真度（批量优化 + 进度条） ----------
    def compute_reconstruction_metrics(self) -> Dict[str, float]:
        """批量编码、解码，逐对比较时显示进度条。"""
        self._log("info", "开始重建保真度计算...")
        # 批量编码
        encodings = self.tokenizer(self.texts, padding=False, truncation=False)
        all_ids = encodings.input_ids
        # 批量解码
        decoded_texts = self.tokenizer.batch_decode(all_ids, skip_special_tokens=False)

        exact_matches = 0
        total_cer = 0.0
        total_chars = 0

        with tqdm(total=len(self.texts), desc="重建对比", unit="text") as pbar:
            for idx, (orig, decoded) in enumerate(zip(self.texts, decoded_texts)):
                if orig == decoded:
                    exact_matches += 1
                total_cer += edit_distance(orig, decoded)
                total_chars += len(orig)
                pbar.update(1)
                # 每 10% 记录一次日志
                if pbar.n % max(1, len(self.texts)//10) == 0 and pbar.n > 0:
                    self._log("info", f"重建对比进度: {pbar.n}/{len(self.texts)} ({100*pbar.n/len(self.texts):.1f}%)")

        exact_match_rate = exact_matches / self.num_texts if self.num_texts > 0 else 0.0
        cer = total_cer / total_chars if total_chars > 0 else 0.0
        self._log("info", f"重建精确匹配率: {exact_match_rate:.4f}, 字符错误率: {cer:.4f}")
        return {
            "reconstruction_exact_match": exact_match_rate,
            "reconstruction_cer": cer,
        }

    # ---------- 分布统计 ----------
    def compute_token_distribution_metrics(self) -> Dict[str, float]:
        if not self.token_freq:
            return {"token_entropy": 0.0, "token_entropy_norm": 0.0, "token_gini": 0.0}

        freq_values = list(self.token_freq.values())
        total = sum(freq_values)
        entropy = 0.0
        for count in freq_values:
            p = count / total
            entropy -= p * math.log2(p)

        actual_vocab = len(freq_values)
        max_entropy = math.log2(actual_vocab) if actual_vocab > 1 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

        sorted_counts = sorted(freq_values)
        n = len(sorted_counts)
        sum_i_ci = 0.0
        for i, count in enumerate(sorted_counts, 1):
            sum_i_ci += i * count
        gini = (2.0 * sum_i_ci / (n * total)) - (n + 1) / n
        return {
            "token_entropy": entropy,
            "token_entropy_norm": entropy_norm,
            "token_gini": gini,
        }

    # ---------- 特殊 Token 测试 ----------
    def test_special_tokens(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, "vocab_size") else 0
        unk_id = self.unk_token_id

        for token_type in ["bos_token", "eos_token", "unk_token", "pad_token"]:
            token_str = self.special_tokens.get(token_type)
            if not token_str:
                result[token_type.split("_")[0]] = {"error": f"未配置 {token_type}"}
                continue

            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            if unk_id is not None:
                exists_in_vocab = (token_id != unk_id)
            else:
                exists_in_vocab = (0 <= token_id < vocab_size) if vocab_size > 0 else True

            encode_ok = False
            decode_ok = False
            error = None
            try:
                encoded_ids = self.tokenizer.encode(token_str)
                if hasattr(encoded_ids, "ids"):
                    first_id = encoded_ids.ids[0] if encoded_ids.ids else None
                else:
                    first_id = encoded_ids[0] if encoded_ids else None
                encode_ok = (first_id == token_id)

                decoded_text = self.tokenizer.decode([token_id])
                decode_ok = token_str in decoded_text or decoded_text.strip() == token_str.strip()
            except Exception as e:
                error = str(e)

            key = token_type.split("_")[0]
            result[key] = {
                "id": token_id,
                "text": token_str,
                "exists_in_vocab": exists_in_vocab,
                "encode_ok": encode_ok,
                "decode_ok": decode_ok,
                "error": error,
            }
        return result

    # ---------- 聊天模板测试 ----------
    def test_chat_template(self) -> Dict[str, Any]:
        if not hasattr(self.tokenizer, "apply_chat_template") or self.tokenizer.chat_template is None:
            return {"enabled": False, "template_available": False, "samples": []}

        samples_output = []
        for sample in tqdm(_CHAT_TEST_SAMPLES, desc="聊天模板测试", unit="样本"):
            messages = sample["messages"]
            try:
                formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                ids_list = self.tokenizer.encode(formatted_text)
                if hasattr(ids_list, "ids"):
                    ids_list = ids_list.ids
                if not isinstance(ids_list, list):
                    ids_list = list(ids_list)
                encoded_length = len(ids_list)

                decoded_text = self.tokenizer.decode(ids_list)
                ids_2 = self.tokenizer.encode(formatted_text)
                if hasattr(ids_2, "ids"):
                    ids_2 = ids_2.ids
                ids_consistent = (ids_list == ids_2)
                roundtrip_ok = (decoded_text == formatted_text) and ids_consistent

                samples_output.append({
                    "messages": messages,
                    "formatted_text": formatted_text,
                    "decoded_text": decoded_text,
                    "encoded_length": encoded_length,
                    "roundtrip_ok": roundtrip_ok,
                    "ids_1": ids_list,
                    "ids_2": ids_2,
                    "ids_consistent": ids_consistent,
                })
            except Exception as e:
                samples_output.append({"messages": messages, "error": str(e)})

        return {"enabled": True, "template_available": True, "samples": samples_output}

    # ---------- 汇总评估 ----------
    def evaluate(self, speed_warmup: int = 1, speed_repeats: int = 3) -> Dict[str, Any]:
        self._log("info", "开始评估...")
        report = {}

        report["num_texts"] = self.num_texts
        report["total_chars"] = self.total_chars
        report["total_tokens"] = self.total_tokens
        report["total_words"] = self.total_words
        self._log("info",
                  f"数据集统计: 文本数={self.num_texts}, 字符数={self.total_chars}, Token数={self.total_tokens}, 单词数={self.total_words}")

        report["compression_ratio"] = self.compute_compression_ratio()
        char_per_token_data = self.compute_char_per_token_by_category()
        report.update(char_per_token_data)
        report["fertility"] = self.compute_fertility()
        self._log("info", f"压缩比={report['compression_ratio']:.4f}, 生育率={report['fertility']:.4f}, "
                          f"每Token字符数(拉丁/CJK/数字/其他): {char_per_token_data.get('char_per_token_latin', 0):.4f}/"
                          f"{char_per_token_data.get('char_per_token_cjk', 0):.4f}/"
                          f"{char_per_token_data.get('char_per_token_digits', 0):.4f}/"
                          f"{char_per_token_data.get('char_per_token_other', 0):.4f}")

        tokens_per_sec, chars_per_sec = self.measure_encoding_speed(warmup=speed_warmup, repeats=speed_repeats)
        report["encoding_speed_tokens_per_sec"] = tokens_per_sec
        report["encoding_speed"] = chars_per_sec
        self._log("info", f"编码速度: {tokens_per_sec:.2f} tokens/秒, {chars_per_sec:.2f} chars/秒")

        report["unk_rate"] = self.compute_unk_rate()
        vocab_data = self.compute_vocab_utilization()
        report.update(vocab_data)
        self._log("info", f"UNK率={report['unk_rate']:.4f}, 词汇表利用率={report['vocab_utilization']:.4f} "
                          f"({report['vocab_size_actual']}/{report['vocab_size_total']})")

        recon_data = self.compute_reconstruction_metrics()
        report.update(recon_data)
        self._log("info",
                  f"重建精确匹配率={report['reconstruction_exact_match']:.4f}, 字符错误率={report['reconstruction_cer']:.4f}")

        dist_data = self.compute_token_distribution_metrics()
        report.update(dist_data)
        self._log("info",
                  f"Token熵={report['token_entropy']:.4f}, 归一化熵={report['token_entropy_norm']:.4f}, 基尼系数={report['token_gini']:.4f}")

        report["special_tokens_test"] = self.test_special_tokens()
        report["chat_template_test"] = self.test_chat_template()
        # 特殊Token和聊天模板测试结果较复杂，简化记录
        special_ok = all(
            info.get("encode_ok") and info.get("decode_ok") for info in report["special_tokens_test"].values() if
            isinstance(info, dict))
        self._log("info", f"特殊Token测试: {'通过' if special_ok else '部分失败'}")

        ct_test = report["chat_template_test"]
        if ct_test.get("enabled"):
            ok_count = sum(1 for s in ct_test.get("samples", []) if s.get("roundtrip_ok"))
            self._log("info", f"聊天模板测试: 可用, 通过样本数={ok_count}/{len(ct_test.get('samples', []))}")
        else:
            self._log("info", "聊天模板测试: 未启用或无模板")

        self._log("info", "评估完成。")
        return report


# ----------------------------------------------------------------------
# 辅助函数与主入口
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Tokenizer 评估工具")
    parser.add_argument("--tokenizer_path", required=True, help="分词器路径或 HuggingFace 模型名称。")
    parser.add_argument("--test_file", required=True, nargs='+', help="一个或多个测试文件/目录路径，支持 .json/.jsonl 文件或目录。")
    parser.add_argument("--output", default="tokenizer_report.json", help="输出 JSON 报告路径（默认: tokenizer_report.json）。")
    parser.add_argument("--speed_warmup", type=int, default=1, help="编码速度测试预热次数（默认: 1）。")
    parser.add_argument("--speed_repeats", type=int, default=3, help="编码速度测试重复次数（默认: 3）。")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志。")
    parser.add_argument("--log_dir", default="./experiments/logs", help="日志目录（LogManager 使用）。")
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化 LogManager
    log_manager = None
    if _HAS_LOGMANAGER and LogManager is not None:
        import logging
        log_manager = LogManager(
            log_dir=args.log_dir,
            tensorboard=False,
            log_file="eval_taichu_tokenizer.log",
            console_level=logging.DEBUG if args.verbose else logging.INFO,
            file_level=logging.DEBUG
        )
        log_manager.info("Tokenizer 评估工具启动")

    try:
        evaluator = TokenizerEvaluator(
            tokenizer_path=args.tokenizer_path,
            test_files=args.test_file,
            log_manager=log_manager,
        )
        report = evaluator.evaluate(
            speed_warmup=args.speed_warmup,
            speed_repeats=args.speed_repeats,
        )

        # 打印关键指标摘要
        print("\n================ Tokenizer 评估报告 ================")
        print(f"文本数量: {report['num_texts']}")
        print(f"总字符数: {report['total_chars']}")
        print(f"总 Token 数: {report['total_tokens']}")
        print(f"总单词数: {report['total_words']}")
        print(f"压缩比: {report['compression_ratio']:.4f}")
        print(f"生育率: {report['fertility']:.4f}")
        print(f"每 Token 字符数 (拉丁): {report['char_per_token_latin']:.4f}")
        print(f"每 Token 字符数 (CJK): {report['char_per_token_cjk']:.4f}")
        print(f"每 Token 字符数 (数字): {report['char_per_token_digits']:.4f}")
        print(f"每 Token 字符数 (其他): {report['char_per_token_other']:.4f}")
        print(f"编码速度: {report['encoding_speed']:.2f} chars/秒")
        print(f"未知词比率: {report['unk_rate']:.4f}")
        print(f"词汇表利用率: {report['vocab_utilization']:.4f} "
              f"({report['vocab_size_actual']}/{report['vocab_size_total']})")
        print(f"重建精确匹配率: {report['reconstruction_exact_match']:.4f}")
        print(f"重建字符错误率: {report['reconstruction_cer']:.4f}")
        print(f"Token 熵: {report['token_entropy']:.4f}")
        print(f"归一化熵: {report['token_entropy_norm']:.4f}")
        print(f"基尼系数: {report['token_gini']:.4f}")
        print("特殊 Token 测试结果:")
        for token_type, info in report["special_tokens_test"].items():
            status = "OK" if info.get("encode_ok") and info.get("decode_ok") else "FAIL"
            print(f"  {token_type}: {status}")
        print("聊天模板测试结果:")
        ct_test = report["chat_template_test"]
        if ct_test["enabled"]:
            for i, sample in enumerate(ct_test["samples"]):
                if "error" in sample:
                    print(f"  样本 {i+1}: ERROR - {sample['error']}")
                else:
                    ok = sample.get("roundtrip_ok", False)
                    consistent = sample.get("ids_consistent", False)
                    print(f"  样本 {i+1}: {'OK' if ok else 'FAIL'} "
                          f"(ID一致性: {'OK' if consistent else 'FAIL'})")
        else:
            print("  未启用或不可用")
        print("====================================================\n")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if log_manager:
            log_manager.info(f"报告已保存至 {args.output}")

    except Exception as e:
        if log_manager:
            log_manager.error(f"评估过程发生错误: {e}", exc_info=True)
        else:
            print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if log_manager:
            log_manager.close()


if __name__ == "__main__":
    main()