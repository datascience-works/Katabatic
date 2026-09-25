# Model Contributions

New models start **experimental**, in `katabatic/experimental/models/`, which has no API stability guarantee. A model is promoted to **supported** once it has a PyPI extra in the root `pyproject.toml`, a registry entry with `supported: True` in [katabatic/models/registry.py](katabatic/models/registry.py), an integration test, and it passes the model promotion contract enforced by [tests/test_model_registry.py](tests/test_model_registry.py). Promotion moves its folder to `katabatic/models/`, whose import paths are covered by semantic versioning. See [docs/EXPERIMENTAL_MODELS.md](docs/EXPERIMENTAL_MODELS.md) for the current supported/experimental split, or check `ModelRegistry.get_supported_models()` directly.

## Contribution Workflow

Please **do not push directly to `main` or `development`** — these branches are protected and reserved for stable, integration-ready code.

### Step 1: Create a Feature Branch

```bash
git checkout -b feature/<model_name>
```

Replace `<model_name>` with the actual name of your model (e.g., `tabddpm`).

### Step 2: Add Your Model

Inside the `katabatic/experimental/models/` directory:

1. Create a new folder for your model:

   ```text
   katabatic/experimental/models/<model_name>/
   ```

2. Within that folder, follow the format used in existing models (e.g. `ganblr`, `ctgan`). Typically including a 3-file format:

   - `__init__.py`
   - `models.py`
   - `README.md`
   - `utils.py` *(optional)*

   All dependencies are to be declared as an extra in the root [pyproject.toml](pyproject.toml) under `[project.optional-dependencies]`, using an extra name matching `<model_name>`:

   ```toml
   [project.optional-dependencies]
   <model_name> = ["some-package>=1.0,<2.0"]
   ```

   Run `poetry lock` afterwards and commit the updated `poetry.lock`.

3. Your model class should **extend** the `Model` base class:

   ```python
   from katabatic.models.base_model import Model
   ```

   Implement the abstract interface (`train`, `sample`). If the model should work with the artifact pipeline (`LocalArtifactStore` / `TrainTestSplitPipeline`), also declare a non-empty `ARTIFACT_STATE_FILES` tuple and override the paired persistence hooks so trained state can be saved and reloaded:

   - `_save_artifact_state(self, artifact_state_dir)` — write your model's state under `artifact_state_dir`. Call `self._maybe_save_artifact_state(artifact_state_dir)` at the end of `train()` (it no-ops when the caller didn't request persistence) rather than checking `artifact_state_dir` yourself.
   - `load_from_ref(cls, store, ref)` — rehydrate a fitted instance from that state. Resolve the path with the base class helpers `cls._require_state_file(store, ref)` (single-file state) or `cls._require_state_dir(store, ref)` (multi-file state) instead of hand-rolling the existence check; both raise `FileNotFoundError` with a consistent message.

   Both hooks default to raising `NotImplementedError` on `Model`, matching each other's shape. See `katabatic/models/ctgan/models.py` for a reference implementation (single-file state) or `katabatic/models/great/models.py` for a multi-file, directory-based one.

4. Follow the API shape shared by every supported model so a user who's used one can guess the others (enforced only loosely — `Model`'s abstract method signatures document it, but subclasses can technically diverge):

   - `train(self, data_dir, *args, synthetic_dir=None, artifact_state_dir=None, **kwargs) -> Self`. `data_dir` is always the first positional parameter (never `dataset_dir`/`dataset`); `synthetic_dir` and `artifact_state_dir` are always named, keyword-only parameters, not values pulled out of `**kwargs`. Always `return self`.
   - `sample(self, n_samples=None, *args, **kwargs)`. The row-count parameter is always named `n_samples` (never `n` or `size`), and should default to `None`, falling back to the number of rows the model was trained on (store that count — e.g. `self._n_train_rows` — during `fit()`/`train()` if you need it).
   - If the model has a natural in-memory fitting step, expose it as `fit(self, X, y=None, ...) -> Self` (sklearn's shape), with `train()` doing the `data_dir` I/O and then calling `self.fit(...)`. If `y` is `None`, `X` is treated as the already-combined frame. Skip this when it doesn't make sense (e.g. TabSyn's preprocessing is inherently file-based); document why in the class docstring rather than forcing it.
   - Don't override `evaluate()`. `Model.evaluate(real_data, ...)` scores any fitted model on the six evaluation dimensions, so every model is comparable. It calls `sample()` in the shape above; the stability dimension also passes a `seed` keyword, so accept one if you want stability runs to be reproducible. If the model has its own diagnostic (a training loss, a model-specific TSTR), expose it under a distinct name such as `evaluate_loss()`. Users who want a different evaluation pass it as `model.evaluate(real_df, pipeline=...)` rather than subclassing.

### Step 3: Register the Model

Add an entry to `ModelRegistry._models` in [katabatic/models/registry.py](katabatic/models/registry.py):

```python
"<model_name>": {
    "module": "katabatic.experimental.models.<model_name>.models",
    "class": "<ModelClassName>",
    "dependencies": ["some-package"],  # importable module names, used for the runtime dependency check
    "extra": "<model_name>",           # must match the pyproject.toml extra and the registry key
    "supported": False,                # new models start unsupported (and in katabatic/experimental/)
},
```

`dataset_requirements` (e.g. `allowed_tasks`) is optional, add it if your model only supports certain task types.

### Step 4: Add Tests

Add an integration smoke test at `tests/test_integration_<model_name>.py`, guarded with `pytest.importorskip(...)` for the model's heavy dependencies and marked with `@pytest.mark.integration` and a model-specific marker (register new markers under `[tool.pytest.ini_options]` in `pyproject.toml`). Follow the pattern in `tests/test_integration_ganblr.py` or `tests/test_integration_ctgan.py`: run the model through `TrainTestSplitPipeline` with a `LocalArtifactStore`, and assert the expected model/synthetic/evaluation artifacts are written.

### Step 5: Push and Open a Pull Request

```bash
git add .
git commit -m "feat: add <model_name> model"
git push origin feature/<model_name>
```

Open a PR into `development` and:

- Include a summary of the model and any new dependencies
- Confirm the model is registered with `supported: False` (promotion is a separate, deliberate step)

## Code Formatting

Formatting and linting run via [pre-commit](.pre-commit-config.yaml) (`ruff-check --fix`, `ruff-format`, plus conventional commit message checks) and are enforced as a required CI job. Install the hooks locally so they run automatically:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Promoting a Model from Experimental to Supported

Promotion is done by a maintainer once a model has proven stable. It requires all of the following:

1. **Test harness**: an integration smoke test exists at `tests/test_integration_<model_name>.py` and runs the artifact pipeline (train → sample → evaluate, with state persisted via `ARTIFACT_STATE_FILES` and reloaded via `load_from_ref`).
2. **Model promotion contract**: `tests/test_model_registry.py::test_model_promotion_contract` passes for the model — it checks that the registry `extra` is non-empty and equals the model name; `module`/`class` are declared and import cleanly; the model class exposes `train`, `sample`, and `load_from_ref`; `ARTIFACT_STATE_FILES` is non-empty; and the integration test file from Step 4 exists. `load_from_ref` must override the base class's default (which raises `NotImplementedError`); `_save_artifact_state` follows the same pattern but isn't separately checked here — it's exercised implicitly whenever the integration test round-trips state through the artifact pipeline.
3. **CI checks are green**: add the model to the `changes` job's `ALL_MODELS` list and its `paths-filter` block in [.github/workflows/ci.yml](.github/workflows/ci.yml), so `integration-<model_name>` runs on relevant changes. That job runs both the integration test and the model promotion contract (step 2 above) as separate steps in one run — no separate YAML wiring needed for the contract check itself.
4. **Dependencies live in the model's own extra**: declared under `[project.optional-dependencies].<model_name>` in the root `pyproject.toml` (not bundled into an unrelated extra), with `poetry.lock` up to date.
5. **Switch to supported status**: once 1–4 are satisfied, move the model's folder with `git mv katabatic/experimental/models/<model_name> katabatic/models/<model_name>` and update imports of the old path (its own modules, tests and `benchmarks/examples/<model_name>/`). In `katabatic/models/registry.py`, change the model's `module` to `katabatic.models.<model_name>.models` and flip `"supported": False` to `"supported": True`; `tests/test_model_registry.py::test_support_status_matches_package_location` fails if the two disagree. Then update the expected set in `tests/test_model_registry.py::test_supported_models_list`, and move the model from the experimental to the supported table in `docs/EXPERIMENTAL_MODELS.md` and the README install matrix. Coverage and the API stability guarantee follow the folder: `katabatic/experimental/*` is omitted from coverage.
