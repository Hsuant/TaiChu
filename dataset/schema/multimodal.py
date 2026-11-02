"""Reference-based multimodal sample schema."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .common import (
    CommonSample,
    binary_mask,
    register_sample,
    require_aligned,
    token_ids,
)


@register_sample
@dataclass(slots=True, kw_only=True)
class MultimodalSample(CommonSample):
    """Text tokens plus modality references and optional flattened tensors."""

    input_ids: list[int]
    image_refs: list[str] = field(default_factory=list)
    image_pixel_values: list[float] = field(default_factory=list)
    image_shape: list[int] = field(default_factory=list)
    modality_mask: list[int] | None = None
    labels: list[int] | None = None
    attention_mask: list[int] | None = None
    modality_metadata: dict[str, Any] = field(default_factory=dict)
    sample_type: ClassVar[str] = "multimodal"
    array_fields: ClassVar[tuple[str, ...]] = (
        "input_ids",
        "image_pixel_values",
        "image_shape",
        "modality_mask",
        "labels",
        "attention_mask",
    )

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        self.input_ids = token_ids(self.input_ids)
        self.image_refs = list(self.image_refs)
        if any(not isinstance(item, str) or not item for item in self.image_refs):
            raise ValueError("image_refs must contain non-empty strings")
        self.image_pixel_values = [float(value) for value in self.image_pixel_values]
        if not all(math.isfinite(value) for value in self.image_pixel_values):
            raise ValueError("image_pixel_values must be finite")
        self.image_shape = token_ids(self.image_shape, name="image_shape") if self.image_shape else []
        if self.image_pixel_values and (
            len(self.image_shape) < 2
            or _product(self.image_shape) != len(self.image_pixel_values)
        ):
            raise ValueError("image_shape must describe image_pixel_values exactly")
        self.modality_mask = (
            None
            if self.modality_mask is None
            else binary_mask(self.modality_mask, name="modality_mask")
        )
        self.labels = (
            None
            if self.labels is None
            else token_ids(self.labels, name="labels", allow_ignore_index=True)
        )
        self.attention_mask = (
            None
            if self.attention_mask is None
            else binary_mask(self.attention_mask, name="attention_mask")
        )
        for name in ("modality_mask", "labels", "attention_mask"):
            value = getattr(self, name)
            if value is not None:
                require_aligned("multimodal sample", self.input_ids, value)


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


__all__ = ["MultimodalSample"]
