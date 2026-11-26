"""Sample from GReaT in chunks, with a time limit and progress messages.

GReaT writes one batch of 100 rows per call to the language model and repeats
until it has enough valid rows. On the Adult dataset that is hundreds of calls
in one uninterrupted loop, with no progress and no limit, which is why a run can
look stuck for hours. Sampling in chunks keeps the model code unchanged and adds:

* a progress line after every chunk, with an estimate of the time left;
* a time limit (default 1 hour). If it is reached, the run stops with a clear
  error instead of running for ever. Following the stream's 2-hour rule, that
  counts as a known limitation of the model, not a fault;
* a different seed for every chunk (``seed``, ``seed + 1``, ...), so chunks do
  not repeat each other, while the whole run stays reproducible.
"""

from time import perf_counter

import pandas as pd

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_MAX_SECONDS = 3600


class SamplingBudgetExceeded(RuntimeError):
    """Sampling used more than its time limit before reaching the row count."""


def sample_in_chunks(
    model,
    n_rows: int,
    device: str,
    seed: int,
    columns=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    clock=perf_counter,
) -> pd.DataFrame:
    """Return ``n_rows`` synthetic rows from ``model.sample``, chunk by chunk."""
    if n_rows <= 0:
        raise ValueError("n_rows must be greater than 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    start = clock()
    parts = []
    total = 0
    chunk_number = 0

    while total < n_rows:
        wanted = min(chunk_size, n_rows - total)
        chunk = pd.DataFrame(
            model.sample(wanted, device=device, seed=seed + chunk_number),
            columns=columns,
        )
        if chunk.empty:
            raise RuntimeError(
                f"GReaT returned no rows after {total} of {n_rows}. "
                "The model may need more epochs or a larger max_length."
            )
        parts.append(chunk)
        total += len(chunk)
        chunk_number += 1

        elapsed = clock() - start
        rate = total / elapsed if elapsed > 0 else 0.0
        left = (n_rows - total) / rate if rate > 0 and total < n_rows else 0.0
        print(
            f"  sampled {min(total, n_rows)}/{n_rows} rows in {elapsed:.0f} s"
            f" (about {left:.0f} s left)",
            flush=True,
        )

        if max_seconds is not None and total < n_rows and elapsed > max_seconds:
            raise SamplingBudgetExceeded(
                f"GReaT sampling stopped after {elapsed:.0f} s with {total} of "
                f"{n_rows} rows (limit {max_seconds:.0f} s). Known limitation "
                "for this dataset, see benchmarks/examples/great/README.md."
            )

    return pd.concat(parts, ignore_index=True).head(n_rows)
