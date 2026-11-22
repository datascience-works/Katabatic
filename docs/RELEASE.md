# Releasing Katabatic

How a new version of `katabatic` gets from `development` to PyPI, and the checks it must pass first.

Owner: PyPI Release & DevOps stream (T3 2026). Product Owner rule: **nothing is published until it has been properly tested.**

## How publishing works today

| Workflow | Trigger | Publishes to | Auth |
|---|---|---|---|
| `.github/workflows/publish-testpypi.yml` | Pre-release tag (`v1.2.0rc1`, `a`, `b`, `.dev`) or manual run | TestPyPI | Trusted Publishing (falls back to `TEST_PYPI_API_TOKEN` if that secret is set) |
| `.github/workflows/publish-pypi.yml` | GitHub Release published from a `vX.Y.Z` tag, or manual run | PyPI | `PYPI_API_TOKEN` secret (Trusted Publishing not set up yet) |

Both workflows check that the tag matches the version in `pyproject.toml`, run `poetry check`, then `poetry build`.

Neither publish workflow runs the test suite. The release gate below is what stops an untested build from shipping.

## Release gate

A version may be published only when every item is ticked. Record the ticked list in the release PR.

### 1. Code and CI
- [ ] Every change for this release is merged into `development` through reviewed PRs.
- [ ] CI is green on the exact commit being released: `lint-and-test` (Ruff, Bandit, pip-audit, fast pytest, core coverage floor, wheel build) and every `integration-<model>` job that ran.
- [ ] The model promotion contract passes for every model whose code changed (`make contract MODEL=<name>`).
- [ ] The `setup-script` job passes on Ubuntu, macOS and Windows.
- [ ] No open issue is labelled as a release blocker.

### 2. Version and changelog
- [ ] The version is bumped in `pyproject.toml` (`poetry version patch|minor|major`), following Semantic Versioning.
- [ ] `CHANGELOG.md` has a section for the new version listing Added / Changed / Fixed, plus upgrade notes for any breaking change.
- [ ] `README.md` and the model docs mention no feature that this version doesn't contain.

### 3. TestPyPI dry run
- [ ] Release candidate tagged (`vX.Y.Zrc1`) and published to TestPyPI by `publish-testpypi.yml`.
- [ ] Installed into a clean Python 3.11 environment from TestPyPI:
  ```bash
  python -m venv /tmp/kt && source /tmp/kt/bin/activate
  pip install --index-url https://test.pypi.org/simple/ \
              --extra-index-url https://pypi.org/simple/ "katabatic==X.Y.Zrc1"
  python -c "import katabatic; print(katabatic.__name__)"
  ```
- [ ] Smoke test passed: train and sample one lightweight supported model on a bundled dataset, then run `evaluate()`.
- [ ] At least one extra installs cleanly, e.g. `pip install "katabatic[ctgan]==X.Y.Zrc1"`.

### 4. Sign-off
- [ ] Testing & Model Validation lead: tests and known-issue fixes are in.
- [ ] Benchmarking & Evaluation lead: no benchmark regression against the previous release.
- [ ] Documentation lead: changelog and docs are updated.
- [ ] Product Owner has approved the release.

## Steps to release

1. Open a release PR into `development`: version bump and changelog. Paste the release gate into the PR description.
2. Once it's merged, tag the release candidate on that commit and push it:
   ```bash
   git tag vX.Y.Zrc1 && git push origin vX.Y.Zrc1
   ```
3. Run the TestPyPI checks in section 3. If anything fails, fix it, merge and tag `rc2`.
4. Merge `development` into `main` through a PR.
5. On GitHub, create a Release from a new `vX.Y.Z` tag on `main`, with the changelog section as the release notes. Publishing the Release runs `publish-pypi.yml`.
6. Check that https://pypi.org/project/katabatic/ shows the new version, and do a clean `pip install katabatic==X.Y.Z`.
7. Sync `main` back into `development` (as done after v1.1.0 in #257).

Don't push a `vX.Y.Z` tag and create the Release separately. Both would try to publish the same version, and PyPI rejects re-uploads.

## Moving PyPI publishing to Trusted Publishing

TestPyPI already uses Trusted Publishing. To switch PyPI over:

1. A PyPI owner of the `katabatic` project adds a trusted publisher: owner `datascience-works`, repository `Katabatic`, workflow `publish-pypi.yml`, environment `pypi`.
2. Remove the `password:` line from the publish step in `publish-pypi.yml`. The workflow already has `id-token: write`.
3. Publish a release to confirm it works, then delete the `PYPI_API_TOKEN` repository secret.

This needs someone with owner access on PyPI and admin access on the GitHub repository.

## Python version policy

Katabatic supports Python 3.11 only (`requires-python = ">=3.11,<3.12"`). Docker base-image updates from Dependabot must stay on `python:3.11-*` until support for newer Python versions has been tested and added.
