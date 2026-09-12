# Model Contributions

New models start **experimental**. A model is promoted to **supported** once it has a PyPI extra in the root `pyproject.toml`, a registry entry with `supported: True` in [katabatic/models/registry.py](katabatic/models/registry.py), an integration test, and it passes the model promotion contract enforced by [tests/test_model_registry.py](tests/test_model_registry.py). See [docs/EXPERIMENTAL_MODELS.md](docs/EXPERIMENTAL_MODELS.md) for the current supported/experimental split, or check `ModelRegistry.get_supported_models()` directly.

## 🛠 Contribution Workflow

Please **do not push directly to `main` or `development`** — these branches are protected and reserved for stable, integration-ready code.

### Step 1: Create a Feature Branch

```bash
git checkout -b feature/<model_name>
```

Replace `<model_name>` with the actual name of your model (e.g., `tabddpm`).

### Step 2: Add Your Model

Inside the `katabatic/models/` directory:

1. Create a new folder for your model:

   ```
   katabatic/models/<model_name>/
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

   Implement the abstract interface (`train`, `evaluate`, `sample`). If the model should work with the artifact pipeline (`LocalArtifactStore` / `TrainTestSplitPipeline`), also implement `load_from_ref` and declare a non-empty `ARTIFACT_STATE_FILES` tuple so trained state can be persisted and reloaded. See `katabatic/models/ctgan/models.py` for a reference implementation.

### Step 3: Register the Model

Add an entry to `ModelRegistry._models` in [katabatic/models/registry.py](katabatic/models/registry.py):

```python
"<model_name>": {
    "module": "katabatic.models.<model_name>.models",
    "class": "<ModelClassName>",
    "dependencies": ["some-package"],  # importable module names, used for the runtime dependency check
    "extra": "<model_name>",           # must match the pyproject.toml extra and the registry key
    "supported": False,                # new models start unsupported
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
- Confirm the model is registered with `supported: False` (promotion is a separate, deliberate step).

## 🧹 Code Formatting

Formatting and linting run via [pre-commit](.pre-commit-config.yaml) (`ruff-check --fix`, `ruff-format`, plus conventional commit message checks) and are enforced as a required CI job. Install the hooks locally so they run automatically:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 🚀 Promoting a Model from Experimental to Supported

Promotion is done by a maintainer once a model has proven stable. It requires all of the following:

1. **Test harness**: an integration smoke test exists at `tests/test_integration_<model_name>.py` and runs the artifact pipeline (train -> sample -> evaluate, with state persisted via `ARTIFACT_STATE_FILES` and reloaded via `load_from_ref`).
2. **Compliance with the model promotion contract (test registry)**: `tests/test_model_registry.py::test_model_promotion_contract` passes for the model. It checks that: the registry `extra` is non-empty and equals the model name; `module`/`class` are declared and import cleanly; the model class exposes `train`, `sample`, and `load_from_ref`; `ARTIFACT_STATE_FILES` is non-empty; and the integration test file from Step 4 exists.
3. **CI checks are green**: the model is added to the `ALL_MODELS` matrix and the `paths-filter` block in [.github/workflows/ci.yml](.github/workflows/ci.yml) so its `integration-<model_name>` job runs on relevant changes, and the `model-contract` job (`poetry install -E all` + `pytest tests/test_model_registry.py`) passes.
4. **Poetry dependencies live in the model's own extra**: dependencies are declared under `[project.optional-dependencies].<model_name>` in the root `pyproject.toml` (not bundled into an unrelated extra), and `poetry.lock` is up to date.
5. **Switch to supported status**: once 1–4 are satisfied, flip `"supported": False` to `"supported": True` for the model in `katabatic/models/registry.py`, update the expected set in `tests/test_model_registry.py::test_supported_models_list`, remove the model's directory from the coverage `omit` list in `pyproject.toml` if present, and update `docs/EXPERIMENTAL_MODELS.md` and the README install matrix to move it out of the experimental table.
