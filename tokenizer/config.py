"""Configuration loading and validation for Tokenizer training."""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from tokenizer.special_tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
)


class ConfigError(ValueError):
    """Raised when the Tokenizer training configuration is invalid."""


@dataclass(frozen=True)
class DataConfig:
    files: Tuple[Path, ...]
    epoch: int = 1
    text_key: str = "text"
    max_text_chars: Optional[int] = None
    skip_empty: bool = True
    num_workers: int = 1
    json_prefix: Optional[str] = None
    allow_full_json_load: bool = False


@dataclass(frozen=True)
class PreTokenizerConfig:
    add_prefix_space: bool = False
    use_regex: bool = True
    digit_split: bool = True


@dataclass(frozen=True)
class NormalizerConfig:
    type: str = "nfc"
    steps: Tuple[Dict[str, str], ...] = ()


@dataclass(frozen=True)
class SpecialTokenConfig:
    ordered_tokens: Tuple[str, ...]
    bos_token: str
    eos_token: str
    pad_token: str
    unk_token: str

    @property
    def additional_special_tokens(self) -> Tuple[str, ...]:
        standard = {self.bos_token, self.eos_token, self.pad_token, self.unk_token}
        return tuple(token for token in self.ordered_tokens if token not in standard)


@dataclass(frozen=True)
class TrainerConfig:
    vocab_size: int = 64000
    min_frequency: int = 3
    show_progress: bool = True
    special_tokens: SpecialTokenConfig = field(
        default_factory=lambda: SpecialTokenConfig(
            ordered_tokens=SPECIAL_TOKENS,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
        )
    )


@dataclass(frozen=True)
class RuntimeConfig:
    monitor_interval_seconds: float = 0.1
    progress_every_records: int = 100000
    full_reload_validation: bool = False


@dataclass(frozen=True)
class SaveConfig:
    directory: Path
    model_max_length: int = 32768
    overwrite: bool = True


@dataclass(frozen=True)
class TokenizerTrainingConfig:
    data: DataConfig
    normalizer: NormalizerConfig
    pre_tokenizer: PreTokenizerConfig
    trainer: TrainerConfig
    runtime: RuntimeConfig
    save: SaveConfig
    model_meta: Dict[str, Any] = field(default_factory=dict)


def load_config_dict(path: Path) -> Dict[str, Any]:
    """Load a YAML file without importing Tokenizers."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"配置文件不存在: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError("配置文件顶层必须是 YAML 映射")
    return config


def parse_config(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
    require_data_files: bool = True,
) -> TokenizerTrainingConfig:
    """Validate a raw mapping and return immutable typed configuration."""
    config_dir = (config_dir or Path.cwd()).resolve()
    data_raw = _mapping(config.get("data"), "data")
    files = _resolve_files(data_raw.get("files"), config_dir, require_data_files)
    data = DataConfig(
        files=files,
        epoch=_positive_int(data_raw.get("epoch", 1), "data.epoch"),
        text_key=_non_empty_str(data_raw.get("text_key", "text"), "data.text_key"),
        max_text_chars=_optional_positive_int(
            data_raw.get("max_text_chars", data_raw.get("max_text_length")),
            "data.max_text_chars",
        ),
        skip_empty=_bool(data_raw.get("skip_empty", True), "data.skip_empty"),
        num_workers=_positive_int(data_raw.get("num_workers", 1), "data.num_workers"),
        json_prefix=_optional_str(data_raw.get("json_prefix"), "data.json_prefix"),
        allow_full_json_load=_bool(
            data_raw.get("allow_full_json_load", False),
            "data.allow_full_json_load",
        ),
    )

    pre_raw = _mapping(config.get("pre_tokenizer"), "pre_tokenizer")
    if pre_raw.get("type", "bytelevel") != "bytelevel":
        raise ConfigError("pre_tokenizer.type 必须为 bytelevel")
    pre_tokenizer = PreTokenizerConfig(
        add_prefix_space=_bool(
            pre_raw.get("add_prefix_space", False),
            "pre_tokenizer.add_prefix_space",
        ),
        use_regex=_bool(pre_raw.get("use_regex", True), "pre_tokenizer.use_regex"),
        digit_split=_bool(pre_raw.get("digit_split", True), "pre_tokenizer.digit_split"),
    )

    normalizer_raw = _mapping(config.get("normalizer", {"type": "nfc"}), "normalizer")
    normalizer_type = _non_empty_str(
        normalizer_raw.get("type", "nfc"),
        "normalizer.type",
    )
    if normalizer_type not in {"none", "nfc", "nfkc", "sequence"}:
        raise ConfigError("normalizer.type 必须是 none、nfc、nfkc 或 sequence")
    raw_steps = normalizer_raw.get("steps", [])
    if not isinstance(raw_steps, list) or not all(isinstance(step, dict) for step in raw_steps):
        raise ConfigError("normalizer.steps 必须是映射列表")
    normalizer = NormalizerConfig(
        type=normalizer_type,
        steps=tuple(dict(step) for step in raw_steps),
    )

    trainer_raw = _mapping(config.get("trainer"), "trainer")
    special_tokens = _code_defined_special_tokens(trainer_raw)
    trainer = TrainerConfig(
        vocab_size=_positive_int(trainer_raw.get("vocab_size", 64000), "trainer.vocab_size"),
        min_frequency=_positive_int(
            trainer_raw.get("min_frequency", 3),
            "trainer.min_frequency",
        ),
        show_progress=_bool(
            trainer_raw.get("show_progress", True),
            "trainer.show_progress",
        ),
        special_tokens=special_tokens,
    )
    if trainer.vocab_size <= len(ByteLevelAlphabet.ALPHABET) + len(special_tokens.ordered_tokens):
        raise ConfigError("trainer.vocab_size 必须大于字节字母表与特殊 Token 的总数")

    runtime_raw = _mapping(config.get("runtime", {}), "runtime")
    runtime = RuntimeConfig(
        monitor_interval_seconds=_positive_float(
            runtime_raw.get("monitor_interval_seconds", 0.1),
            "runtime.monitor_interval_seconds",
        ),
        progress_every_records=_positive_int(
            runtime_raw.get("progress_every_records", 100000),
            "runtime.progress_every_records",
        ),
        full_reload_validation=_bool(
            runtime_raw.get("full_reload_validation", False),
            "runtime.full_reload_validation",
        ),
    )

    save_raw = _mapping(config.get("save"), "save")
    if "chat_template_file" in save_raw:
        raise ConfigError(
            "Chat Template 必须由 tokenizer.template.CHAT_TEMPLATE 代码定义，"
            "不能从 save.chat_template_file 导入"
        )
    save_directory = _resolve_path(save_raw.get("directory"), config_dir, "save.directory")
    save = SaveConfig(
        directory=save_directory,
        model_max_length=_positive_int(
            save_raw.get("model_max_length", 32768),
            "save.model_max_length",
        ),
        overwrite=_bool(save_raw.get("overwrite", True), "save.overwrite"),
    )

    model_meta = config.get("model_meta", {})
    if not isinstance(model_meta, dict):
        raise ConfigError("model_meta 必须是映射")

    return TokenizerTrainingConfig(
        data=data,
        normalizer=normalizer,
        pre_tokenizer=pre_tokenizer,
        trainer=trainer,
        runtime=runtime,
        save=save,
        model_meta=dict(model_meta),
    )


def load_config(
    path: Path,
    require_data_files: bool = True,
) -> TokenizerTrainingConfig:
    raw = load_config_dict(path)
    return parse_config(
        raw,
        path.resolve().parent,
        require_data_files=require_data_files,
    )


def _code_defined_special_tokens(trainer_raw: Dict[str, Any]) -> SpecialTokenConfig:
    config_keys = {
        "special_tokens",
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
    }
    imported_keys = sorted(config_keys.intersection(trainer_raw))
    if imported_keys:
        raise ConfigError(
            "特殊 Token 必须由 tokenizer.special_tokens 代码定义，"
            f"不能从 YAML 配置导入: {imported_keys}"
        )
    return SpecialTokenConfig(
        ordered_tokens=SPECIAL_TOKENS,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
    )


def _resolve_files(
    raw_files: Any,
    config_dir: Path,
    require_exists: bool,
) -> Tuple[Path, ...]:
    if isinstance(raw_files, (str, Path)):
        raw_files = [raw_files]
    if not isinstance(raw_files, Iterable) or isinstance(raw_files, (bytes, dict)):
        raise ConfigError("data.files 必须是文件路径列表")

    resolved: List[Path] = []
    for raw_path in raw_files:
        path = _resolve_path(raw_path, config_dir, "data.files")
        if any(char in str(path) for char in "*?[]"):
            matches = sorted(Path(match) for match in glob.glob(str(path), recursive=True))
        elif path.is_dir():
            matches = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".ndjson"}
            )
        else:
            matches = [path]
        resolved.extend(item.resolve() for item in matches if item.is_file())

    if not resolved and require_exists:
        raise ConfigError("data.files 未匹配到任何存在的 JSON/JSONL 文件")
    if not resolved:
        for raw_path in raw_files:
            resolved.append(_resolve_path(raw_path, config_dir, "data.files"))
    unique = tuple(dict.fromkeys(resolved))
    unsupported = [path for path in unique if path.suffix.lower() not in {".json", ".jsonl", ".ndjson"}]
    if unsupported:
        raise ConfigError(f"不支持的训练文件扩展名: {[str(path) for path in unsupported]}")
    return unique


def _resolve_path(raw_path: Any, config_dir: Path, field_name: str) -> Path:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        raise ConfigError(f"{field_name} 必须是非空路径")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _mapping(raw: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{field_name} 必须是映射")
    return raw


def _non_empty_str(raw: Any, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{field_name} 必须是非空字符串")
    return raw


def _optional_str(raw: Any, field_name: str) -> Optional[str]:
    if raw is None:
        return None
    return _non_empty_str(raw, field_name)


def _bool(raw: Any, field_name: str) -> bool:
    if not isinstance(raw, bool):
        raise ConfigError(f"{field_name} 必须是布尔值")
    return raw


def _positive_int(raw: Any, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ConfigError(f"{field_name} 必须是正整数")
    return raw


def _optional_positive_int(raw: Any, field_name: str) -> Optional[int]:
    if raw is None:
        return None
    return _positive_int(raw, field_name)


def _positive_float(raw: Any, field_name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        raise ConfigError(f"{field_name} 必须是正数")
    return float(raw)


class ByteLevelAlphabet:
    """Lazy access to the ByteLevel alphabet without importing at module load."""

    ALPHABET: Tuple[str, ...] = tuple(
        chr(code) for code in range(256)
    )
