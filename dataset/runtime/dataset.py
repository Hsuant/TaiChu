"""Map-style dataset facade over immutable indexed storage."""
from __future__ import annotations

from torch.utils.data import Dataset

from ..schema import CommonSample
from ..storage import IndexedSampleReader


class IndexedTokenDataset(Dataset[CommonSample]):
    def __init__(self, reader: IndexedSampleReader) -> None:
        self.reader = reader

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> CommonSample:
        return self.reader[index]

    @property
    def dataset_id(self) -> str:
        return self.reader.dataset_id

    @property
    def version(self) -> str:
        return self.reader.version

    @property
    def sample_type(self) -> str:
        return self.reader.sample_type


__all__ = ["IndexedTokenDataset"]
