"""SFT 离线数据构建配置。

该配置只描述通用 SFT 构建语义，不包含任何平台或数据集专用字段。输入必须是
canonical ``messages`` 记录；输出固定包含：

``build_manifest.json``
    本次构建的配额、来源、Token 和划分统计。
``source/``
    清洗、去重和配额选择后的可读 JSONL。
``train/`` and ``validation/``
    由紧凑 token cache 直接生成的训练与验证 Parquet Dataset。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RawSourceConfig:
    """One format-specific raw source adapter."""

    name: str
    adapter: str
    path: Path
    group: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.adapter or not self.group:
            raise ValueError("raw source name, adapter and group are required")
        if not isinstance(self.path, Path):
            raise TypeError("raw source path must be a pathlib.Path")
        if not isinstance(self.options, Mapping):
            raise TypeError("raw source options must be a mapping")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True, slots=True)
class SFTBuildConfig:
    """Immutable configuration shared by the source and unified build stages."""

    # Tokenizer、最终输出目录和 canonical messages 输入路径。
    tokenizer_path: Path
    output_dir: Path

    # Dataset 身份。训练恢复和实验追踪依赖这两个字段。
    dataset_id: str
    version: str = "1"
    profile: str = "default"

    inputs: tuple[Path, ...] = ()

    # Raw adapters and grouped source caps are optional alternatives to canonical
    # JSONL/Parquet inputs.
    raw_sources: tuple[RawSourceConfig, ...] = ()
    source_quotas: dict[str, int] | None = None

    # 单条 conversation 的完整 Token 上限。超过上限的记录不会进入 tokenize。
    max_seq_length: int = 2048

    # 去重和配额选择后的稳定 train/validation 划分比例。
    validation_ratio: float = 0.01
    seed: int = 42

    # 可选类别配额，例如 {"general": 10000, "math": 5000}。
    # 为 None 时不限制类别数量，但仍有长度校验和内容去重。
    quotas: dict[str, int] | None = None

    # source 层字符过滤边界，用于尽早丢弃空样本和异常超长文本。
    min_chars: int = 1
    max_chars: int = 200_000
    min_user_chars: int = 8
    min_assistant_tokens: int = 1
    max_assistant_tokens: int = 512
    reasoning_sample_ratio: float = 0.0
    drop_system: bool = False

    # 这些参数决定 assistant-only loss 和格式化 EOS 监督。
    assistant_only_loss: bool = True
    train_on_eos: bool = True

    # unified Parquet Dataset 的 shard 与预处理批大小。
    shard_size: int = 50_000
    preprocess_batch_size: int = 1024

    # 特殊 Token 必须与 tokenizer 和训练/推理配置完全一致。
    pad_token: str = "<|pad|>"
    bos_token: str = "<|taichu_start|>"
    eos_token: str = "<|taichu_end|>"

    def __post_init__(self) -> None:
        """在进入耗时的数据扫描前拒绝无效配置。"""
        if not self.inputs and not self.raw_sources:
            raise ValueError("sft requires inputs or raw_sources")
        if not self.profile:
            raise ValueError("sft.profile cannot be empty")
        if not self.dataset_id or not self.version:
            raise ValueError("sft.dataset_id and version must be non-empty")
        if self.max_seq_length < 2:
            raise ValueError("sft.max_seq_length must be >= 2")
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("sft.validation_ratio must be in [0, 1)")
        if self.seed < 0:
            raise ValueError("sft.seed cannot be negative")
        if self.min_chars < 0 or self.max_chars < self.min_chars:
            raise ValueError("sft min/max chars are invalid")
        if self.min_user_chars < 0:
            raise ValueError("sft role character limits cannot be negative")
        if self.min_assistant_tokens <= 0:
            raise ValueError("sft.min_assistant_tokens must be positive")
        if self.max_assistant_tokens < self.min_assistant_tokens:
            raise ValueError(
                "sft.max_assistant_tokens must be >= min_assistant_tokens"
            )
        if not 0.0 <= self.reasoning_sample_ratio <= 1.0:
            raise ValueError("sft.reasoning_sample_ratio must be in [0, 1]")
        if self.shard_size <= 0 or self.preprocess_batch_size <= 0:
            raise ValueError("sft shard/batch sizes must be positive")
        if self.quotas is not None:
            if not self.quotas:
                raise ValueError("sft.quotas cannot be empty")
            if any(not name or count <= 0 for name, count in self.quotas.items()):
                raise ValueError("sft quota names and counts must be positive")
        if self.source_quotas is not None:
            if not self.source_quotas:
                raise ValueError("sft.source_quotas cannot be empty")
            if any(
                not name or count <= 0
                for name, count in self.source_quotas.items()
            ):
                raise ValueError(
                    "sft source quota names and counts must be positive"
                )
        names = [source.name for source in self.raw_sources]
        if len(names) != len(set(names)):
            raise ValueError("sft raw source names must be unique")


def load_sft_build_config(
    path: str | Path,
    *,
    profile: str | None = None,
) -> SFTBuildConfig:
    """加载并严格校验 YAML 配置，禁止未知字段和隐式兼容。"""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load SFT build configs") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("SFT build config root must be a mapping")
    values = dict(raw)
    profiles = values.pop("profiles", None)
    requested_profile = profile or values.pop("profile", None)
    if profiles is not None:
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("sft.profiles must be a non-empty mapping")
        if requested_profile is None:
            raise ValueError("sft.profile is required when profiles are defined")
        try:
            selected_profile = profiles[str(requested_profile)]
        except KeyError as exc:
            raise ValueError(
                f"unknown SFT profile {requested_profile!r}"
            ) from exc
        if not isinstance(selected_profile, dict):
            raise ValueError("selected SFT profile must be a mapping")
        selected_profile = dict(selected_profile)
        output_subdir = selected_profile.pop("output_subdir", requested_profile)
        values.update(selected_profile)
        base_output = Path(values["output_dir"])
        values["output_dir"] = base_output / str(output_subdir)
        values["profile"] = str(requested_profile)
    else:
        values.setdefault("profile", "default")
    for name in ("tokenizer_path", "output_dir"):
        if name not in values:
            raise ValueError(f"missing SFT build config key: {name}")
        values[name] = Path(values[name])
    raw_inputs = values.pop("inputs", [])
    if not isinstance(raw_inputs, list):
        raise ValueError("sft.inputs must be a list")
    values["inputs"] = tuple(Path(item) for item in raw_inputs)
    raw_sources = values.pop("raw_sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("sft.raw_sources must be a list")
    parsed_sources = []
    allowed_source_keys = {"name", "adapter", "path", "group", "options"}
    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("each sft raw source must be a mapping")
        unknown_source = set(item) - allowed_source_keys
        if unknown_source:
            raise ValueError(
                f"unknown SFT raw source keys: {sorted(unknown_source)}"
            )
        parsed_sources.append(
            RawSourceConfig(
                name=str(item["name"]),
                adapter=str(item["adapter"]),
                path=Path(item["path"]),
                group=str(item["group"]),
                options=item.get("options", {}),
            )
        )
    values["raw_sources"] = tuple(parsed_sources)
    if "quotas" in values and values["quotas"] is not None:
        raw_quotas = values["quotas"]
        if not isinstance(raw_quotas, dict):
            raise ValueError("sft.quotas must be a mapping")
        values["quotas"] = {
            str(name): int(count) for name, count in raw_quotas.items()
        }
    if "source_quotas" in values and values["source_quotas"] is not None:
        raw_source_quotas = values["source_quotas"]
        if not isinstance(raw_source_quotas, dict):
            raise ValueError("sft.source_quotas must be a mapping")
        values["source_quotas"] = {
            str(name): int(count)
            for name, count in raw_source_quotas.items()
        }
    allowed = set(SFTBuildConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown SFT build config keys: {sorted(unknown)}")
    return SFTBuildConfig(**values)


__all__ = ["RawSourceConfig", "SFTBuildConfig", "load_sft_build_config"]
