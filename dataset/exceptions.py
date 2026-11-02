"""Exceptions raised by the TaiChu streaming dataset pipeline."""


class TaiChuDatasetError(Exception):
    """Base exception for dataset configuration, reads, and validation."""


class TaiChuConfigurationError(TaiChuDatasetError, ValueError):
    """Raised when dataset or DataLoader settings are invalid."""


class TaiChuDataFormatError(TaiChuDatasetError, ValueError):
    """Raised when a source record or tokenized sample violates the contract."""


__all__ = [
    "TaiChuDatasetError",
    "TaiChuConfigurationError",
    "TaiChuDataFormatError",
]
