"""Common sample contracts and dependency-light validation helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, Mapping

from ..exceptions import TaiChuDataFormatError


SAMPLE_REGISTRY: dict[str, type["CommonSample"]] = {}


def _as_sequence(value: Any, name: str) -> list:
    if value is None:
        raise TaiChuDataFormatError(f"{name} cannot be None")
    if isinstance(value, (str, bytes)):
        raise TaiChuDataFormatError(f"{name} must be a numeric sequence")
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TaiChuDataFormatError(
            f"{name} must be a list-like sequence, got {type(value).__name__}"
        )
    return list(value)


def token_ids(
    value: Any,
    *,
    name: str = "input_ids",
    vocab_size: int | None = None,
    allow_ignore_index: bool = False,
) -> list[int]:
    result: list[int] = []
    for item in _as_sequence(value, name):
        if not isinstance(item, Integral) or isinstance(item, bool):
            raise TaiChuDataFormatError(f"{name} must contain integers only")
        token = int(item)
        if allow_ignore_index and token == -100:
            result.append(token)
            continue
        if token < 0 or (vocab_size is not None and token >= vocab_size):
            raise TaiChuDataFormatError(f"{name} contains invalid token ID {token}")
        result.append(token)
    return result


def binary_mask(value: Any, *, name: str = "loss_mask") -> list[int]:
    if value is None:
        return []
    result = token_ids(value, name=name)
    if any(item not in (0, 1) for item in result):
        raise TaiChuDataFormatError(f"{name} must contain only 0 and 1")
    return result


def float_values(value: Any, *, name: str) -> list[float]:
    result: list[float] = []
    for item in _as_sequence(value, name):
        if not isinstance(item, Real) or isinstance(item, bool):
            raise TaiChuDataFormatError(f"{name} must contain numeric values only")
        number = float(item)
        if not math.isfinite(number):
            raise TaiChuDataFormatError(f"{name} must contain finite values")
        result.append(number)
    return result


def require_aligned(name: str, *sequences: list) -> int:
    lengths = {len(values) for values in sequences}
    if len(lengths) != 1:
        raise TaiChuDataFormatError(
            f"{name} sequences must have identical lengths, got {sorted(lengths)}"
        )
    return lengths.pop() if lengths else 0


def positive_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaiChuDataFormatError(f"{name} must be a non-empty string")
    return value


@dataclass(slots=True, kw_only=True)
class CommonSample:
    """Fields shared by every offline or online training sample."""

    sample_id: str
    dataset_id: str = "default"
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_type: ClassVar[str] = "common"
    array_fields: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        positive_identifier(self.sample_id, "sample_id")
        positive_identifier(self.dataset_id, "dataset_id")
        if not isinstance(self.source_id, str):
            raise TaiChuDataFormatError("source_id must be a string")
        if not isinstance(self.metadata, dict):
            raise TaiChuDataFormatError("metadata must be a dictionary")

    def validate(self) -> None:
        return None


def register_sample(cls: type[CommonSample]) -> type[CommonSample]:
    sample_type = getattr(cls, "sample_type", "")
    if not sample_type or sample_type == "common":
        raise ValueError("registered samples need a unique sample_type")
    existing = SAMPLE_REGISTRY.get(sample_type)
    if existing is not None and existing is not cls:
        raise ValueError(f"sample type {sample_type!r} is already registered")
    SAMPLE_REGISTRY[sample_type] = cls
    return cls


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata value of type {type(value).__name__} is not serializable")


def sample_to_storage(sample: CommonSample) -> tuple[dict[str, Any], dict[str, list]]:
    """Split a typed sample into JSON metadata and binary numeric arrays."""
    if not is_dataclass(sample):
        raise TypeError("sample must be a dataclass instance")
    sample_type = getattr(sample, "sample_type", "")
    if sample_type not in SAMPLE_REGISTRY:
        raise TaiChuDataFormatError(f"unregistered sample type {sample_type!r}")
    sample.validate()

    metadata: dict[str, Any] = {"_sample_type": sample_type}
    array_names = set(getattr(sample, "array_fields", ()))
    for item in fields(sample):
        if item.name == "metadata":
            continue
        value = getattr(sample, item.name)
        if item.name in array_names:
            metadata[f"_array_{item.name}"] = True
            continue
        metadata[item.name] = _jsonable(value)
    metadata["metadata"] = _jsonable(sample.metadata)

    arrays: dict[str, list] = {}
    for name in array_names:
        value = getattr(sample, name, None)
        if value is not None:
            arrays[name] = list(_as_sequence(value, name))
        else:
            metadata[f"_null_{name}"] = True
    return metadata, arrays


def sample_from_storage(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, list],
) -> CommonSample:
    sample_type = metadata.get("_sample_type")
    cls = SAMPLE_REGISTRY.get(sample_type)
    if cls is None:
        raise TaiChuDataFormatError(f"unknown stored sample type {sample_type!r}")
    values = {
        key: value
        for key, value in metadata.items()
        if not key.startswith("_") and key != "metadata"
    }
    values["metadata"] = dict(metadata.get("metadata", {}))
    known = {item.name for item in fields(cls)}
    unknown = set(arrays) - known
    if unknown:
        raise TaiChuDataFormatError(f"unknown stored arrays for {sample_type}: {sorted(unknown)}")
    values.update(arrays)
    sample = cls(**values)
    sample.validate()
    return sample


def get_sample_class(sample_type: str) -> type[CommonSample]:
    try:
        return SAMPLE_REGISTRY[sample_type]
    except KeyError as exc:
        raise TaiChuDataFormatError(f"unknown sample type {sample_type!r}") from exc


__all__ = [
    "CommonSample",
    "SAMPLE_REGISTRY",
    "binary_mask",
    "float_values",
    "get_sample_class",
    "positive_identifier",
    "register_sample",
    "require_aligned",
    "sample_from_storage",
    "sample_to_storage",
    "token_ids",
]
