import json
import random
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tokenizer.config import ConfigError, parse_config
from tokenizer.corpus import CorpusIterator
from tokenizer.special_tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
)
from tokenizer.template import CHAT_TEMPLATE
from tokenizer.tokenizer import TaiChuTokenizerTrainer
from tokenizer.validator import validate_tokenizer


def _write_jsonl(path: Path) -> None:
    rng = random.Random(20261025)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    unique_words = [
        "".join(rng.choices(alphabet, k=7))
        for _ in range(3000)
    ]
    zh = ["人工智能", "分词器", "内存", "模型训练", "中英混合", "性能优化"]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(4000):
            words = rng.choices(unique_words, k=8)
            text = (
                f"{' '.join(words)} "
                f"{''.join(rng.choices(zh, k=4))} "
                f"number={index:08d}\n"
                "def func(x):\n    return x + 1\n"
            )
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _raw_config(data_path: Path, output_dir: Path) -> dict:
    return {
        "data": {
            "files": [str(data_path)],
            "epoch": 1,
            "text_key": "text",
            "max_text_length": 10000,
            "skip_empty": True,
            "num_workers": 1,
        },
        "normalizer": {"type": "nfc"},
        "pre_tokenizer": {
            "type": "bytelevel",
            "add_prefix_space": False,
            "use_regex": True,
            "digit_split": True,
        },
        "trainer": {
            "vocab_size": 512,
            "min_frequency": 1,
            "show_progress": False,
        },
        "runtime": {
            "monitor_interval_seconds": 0.02,
            "progress_every_records": 100000,
        },
        "save": {
            "directory": str(output_dir),
            "model_max_length": 32768,
            "overwrite": True,
        },
        "model_meta": {"name": "tokenizer-test"},
    }


@pytest.fixture(scope="module")
def trained_tokenizer_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tokenizer-training")
    data_path = directory / "corpus.jsonl"
    output_dir = directory / "tokenizer"
    _write_jsonl(data_path)
    config = parse_config(_raw_config(data_path, output_dir), directory)
    trainer = TaiChuTokenizerTrainer(config)
    report = trainer.run()
    return output_dir, report


def test_training_save_reload_and_special_token_ids(trained_tokenizer_dir):
    output_dir, report = trained_tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True)

    assert report["actual_vocab_size"] == 512
    assert len(tokenizer) == 512
    assert report["validation"]["special_token_ids"] == {
        token: index for index, token in enumerate(SPECIAL_TOKENS)
    }
    assert tokenizer.bos_token_id == 3
    assert tokenizer.eos_token_id == 4
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token == BOS_TOKEN
    assert tokenizer.eos_token == EOS_TOKEN
    assert tokenizer.pad_token == PAD_TOKEN
    assert tokenizer.unk_token == UNK_TOKEN
    assert set(tokenizer.get_added_vocab()) == set(SPECIAL_TOKENS)
    assert tokenizer.chat_template == CHAT_TEMPLATE
    assert (output_dir / "chat_template.jinja").read_text(encoding="utf-8") == CHAT_TEMPLATE
    for token_id, token in enumerate(SPECIAL_TOKENS):
        assert tokenizer.encode(token, add_special_tokens=False) == [token_id]
        assert tokenizer.decode([token_id], skip_special_tokens=False) == token

    full_report = validate_tokenizer(output_dir, SPECIAL_TOKENS, 512)
    assert full_report["mode"] == "transformers_full"
    assert full_report["vocab_size"] == 512


@pytest.mark.parametrize(
    "text",
    [
        "人工智能改变了世界。",
        "Byte-level BPE keeps spaces and\nnewlines stable.",
        "中英混合 mixed text, numbers 1234567890 and punctuation!?",
        "def tokenize(text):\n    return text.encode('utf-8')\n",
        "Emoji: 😀🚀 中文标点：，。！？",
        "   leading and trailing spaces   ",
        "empty newline\n\nand tabs\t\t",
    ],
)
def test_text_roundtrip(trained_tokenizer_dir, text):
    output_dir, _ = trained_tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids, skip_special_tokens=False) == text


def test_empty_string_and_long_text(trained_tokenizer_dir):
    output_dir, _ = trained_tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True)
    assert tokenizer.encode("", add_special_tokens=False) == []
    text = "中文 English\n" * 2000
    assert tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=False),
        skip_special_tokens=False,
    ) == text


def test_chat_template_single_and_multi_turn(trained_tokenizer_dir):
    output_dir, _ = trained_tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True)

    generated = tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert generated == (
        "<|taichu_start|>user\n你好<|taichu_end|>\n"
        "<|taichu_start|>assistant\n"
    )

    multi_turn = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答", "reasoning_content": "思考"},
            {"role": "user", "content": "第二问"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert multi_turn == (
        "<|taichu_start|>system\n系统提示<|taichu_end|>\n"
        "<|taichu_start|>user\n第一问<|taichu_end|>\n"
        "<|taichu_start|>assistant\n"
        "<|think|>思考<|/think|>\n第一答<|taichu_end|>\n"
        "<|taichu_start|>user\n第二问<|taichu_end|>\n"
    )


def test_jsonl_streaming_and_diagnostics(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"text":"第一行"}\n'
        "\n"
        '{"text":""}\n'
        '{"other":1}\n'
        "{not-json}\n"
        '{"text":"second line"}\n',
        encoding="utf-8",
    )
    iterator = CorpusIterator([path], max_text_length=5)
    assert list(iterator) == ["第一行", "secon"]
    assert iterator.stats.records_read == 4
    assert iterator.stats.records_emitted == 2
    assert iterator.stats.invalid_records == 1
    assert iterator.stats.missing_text == 1
    assert iterator.stats.empty_text == 1
    assert iterator.stats.truncated_text == 1


def test_json_array_and_object_streaming(tmp_path):
    array_path = tmp_path / "array.json"
    array_path.write_text(
        json.dumps([{"text": "一"}, {"text": "二"}, {"text": "三"}], ensure_ascii=False),
        encoding="utf-8",
    )
    assert list(CorpusIterator([array_path])) == ["一", "二", "三"]

    object_path = tmp_path / "object.json"
    object_path.write_text(
        json.dumps({"records": [{"text": "alpha"}, {"text": "beta"}]}),
        encoding="utf-8",
    )
    assert list(
        CorpusIterator(
            [object_path],
            json_prefix="records",
            allow_full_json_load=True,
        )
    ) == ["alpha", "beta"]


def test_config_rejects_external_special_token_protocol(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text":"x"}\n', encoding="utf-8")
    config = _raw_config(data_path, tmp_path / "out")
    config["trainer"]["special_tokens"] = ["<unk>"]
    with pytest.raises(ConfigError, match="特殊 Token.*代码定义"):
        parse_config(config, tmp_path)


def test_config_rejects_external_chat_template(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text":"x"}\n', encoding="utf-8")
    config = _raw_config(data_path, tmp_path / "out")
    config["save"]["chat_template_file"] = "template.jinja"
    with pytest.raises(ConfigError, match="Chat Template.*代码定义"):
        parse_config(config, tmp_path)
