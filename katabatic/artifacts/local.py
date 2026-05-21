from __future__ import annotations

import json
from pathlib import Path

from katabatic.artifacts.base import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def _full(self, path: str) -> Path:
        return self.root / path

    def save_json(self, path: str, data: dict) -> None:
        full = self._full(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_json(self, path: str) -> dict:
        return json.loads(self._full(path).read_text(encoding="utf-8"))

    def save_bytes(self, path: str, data: bytes) -> None:
        full = self._full(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    def open_path(self, path: str) -> Path:
        return self._full(path)

    def exists(self, path: str) -> bool:
        return self._full(path).exists()
