from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def save_json(self, path: str, data: dict) -> None:
        ...

    @abstractmethod
    def load_json(self, path: str) -> dict:
        ...

    @abstractmethod
    def save_bytes(self, path: str, data: bytes) -> None:
        ...

    @abstractmethod
    def open_path(self, path: str) -> Path:
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        ...
