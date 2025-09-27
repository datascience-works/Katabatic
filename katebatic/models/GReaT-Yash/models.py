from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Project base interface
from katebatic.models.base_model import Model

# Your GReaT implementation (exports GReaT from .great)
# Works whether you expose GReaT at package root or module path
try:
    from be_great import GReaT  # __init__ -> from .great import GReaT
except ImportError:
    from be_great.great import GReaT  # fallback import

from .utils import postprocess_df, log_cuda_info


@dataclass
class GreatConfig:
    # Training
    llm: str = "gpt2"
    epochs: int = 100
    batch_size: int = 8
    efficient_finetuning: str = ""   # "", "lora", ...
    float_precision: Optional[int] = None
    report_to: list[str] = None      # e.g., ["none"]; None => disabled

    # Sampling
    device: str = "cuda"
    guided_sampling: bool = False
    random_feature_order: bool = True
    max_length: int = 512
    k: int = 100
    temperature: float = 0.7

    # Misc
    random_state: int = 42
    random_conditional_col: bool = True  # use your epoch-wise random conditional callback


class GreatModel(Model):
    """
    Thin wrapper around your be_great.GReaT with the project Model API:
      - fit(df): fine-tunes the LLM on the full table
      - generate(n, conditions=None): samples n rows; if conditions are given,
        we sample extra then post-filter to honor exact matches
      - save/load: persist wrapper cfg + call GReaT's own save/load
    """

    def __init__(self, **kwargs: Any) -> None:
        cfg_dict = kwargs.pop("cfg", None) or kwargs
        if cfg_dict.get("report_to") is None:
            cfg_dict["report_to"] = []
        self.cfg = GreatConfig(**cfg_dict)

        self._rng = np.random.RandomState(self.cfg.random_state)
        self._great: Optional[GReaT] = None
        self._is_fit = False

    # ---- required by base class ----
    def fit(self, train_df: pd.DataFrame, **kwargs) -> "GreatModel":
        assert isinstance(train_df, pd.DataFrame), "fit() expects a pandas DataFrame"

        log_cuda_info()  # optional: prints CUDA info if torch is available

        self._great = GReaT(
            llm=self.cfg.llm,
            experiment_dir=kwargs.get("experiment_dir", "trainer_great"),
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            efficient_finetuning=self.cfg.efficient_finetuning,
            float_precision=self.cfg.float_precision,
            report_to=self.cfg.report_to,
            # any extra HF TrainingArguments via train_kwargs:
            **{k: v for k, v in kwargs.items() if k not in ("experiment_dir",)}
        )
        # Use your built-in random conditional column logic during training
        self._great.fit(
            train_df,
            resume_from_checkpoint=False,
            random_conditional_col=self.cfg.random_conditional_col,
        )
        self._is_fit = True
        return self

    def generate(
        self,
        num_rows: int,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._is_fit or self._great is None:
            raise RuntimeError("Call fit() before generate().")

        guided = kwargs.pop("guided_sampling", self.cfg.guided_sampling)
        # Over-sample if we need to satisfy conditions by post-filtering
        n = int(num_rows if not conditions else max(num_rows * 5, num_rows + 500))

        samples = self._great.sample(
            n_samples=n,
            device=self.cfg.device,
            guided_sampling=guided,
            random_feature_order=self.cfg.random_feature_order,
            max_length=self.cfg.max_length,
            k=self.cfg.k,
            temperature=self.cfg.temperature,
            **kwargs,
        )

        # Exact-match post filter for simple conditions (col == value)
        if conditions:
            mask = np.ones(len(samples), dtype=bool)
            for col, val in conditions.items():
                if col in samples.columns:
                    mask &= (samples[col] == val)
            filtered = samples[mask]
            if len(filtered) >= num_rows:
                out = filtered.head(num_rows).reset_index(drop=True)
            else:
                need = num_rows - len(filtered)
                tail = samples[~mask].head(need)
                out = pd.concat([filtered, tail], axis=0).reset_index(drop=True)
        else:
            out = samples.head(num_rows).reset_index(drop=True)

        return postprocess_df(out)

    def save(self, path: str) -> None:
        if self._great is None:
            raise RuntimeError("Nothing to save; fit() the model first.")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Save wrapper config
        (p / "wrapper_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        # Save your GReaT (weights + config.json) into subdir
        self._great.save(str(p / "great"))

    def load(self, path: str) -> "GreatModel":
        p = Path(path)
        cfg_path = p / "wrapper_config.json"
        if cfg_path.exists():
            self.cfg = GreatConfig(**json.loads(cfg_path.read_text()))
        # Restore GReaT internals
        self._great = GReaT.load_from_dir(str(p / "great"))
        self._is_fit = True
        return self
