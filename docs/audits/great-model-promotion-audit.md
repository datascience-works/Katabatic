# GReaT Model Promotion Audit

## Passed Checks

- GReaT registry entry exists.
- `supported: True` is configured.
- GReaT Poetry extra is present.
- Integration tests pass.
- GReaT is included in the CI matrix.
- Registry smoke test successfully creates the GReaT model.
- Ruff validation passes.

## Gaps Identified

- `katabatic/models/great/__init__.py` is missing.
- A model-level `README.md` is missing.
- No GReaT-specific `load_from_ref` implementation was found.

## Test Evidence

Integration test result:

- 2 passed
- 1 skipped
- 11 deselected

Ruff result:

- All checks passed.
