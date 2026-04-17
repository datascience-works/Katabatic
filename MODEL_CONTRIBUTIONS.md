# Model Contributions

## 🛠 Contribution Workflow

To contribute a new model to this project, please **do not push directly to `main` or `development` branches**. These are protected and reserved for stable and integration-ready code.

Follow the steps below to add your model in a structured and maintainable way:

---

## 🔀 Step 1: Create a Feature Branch

Start by creating a new branch off `development`:

```bash
git checkout -b feature/<model_name>
```

Replace `<model_name>` with the actual name of your model (e.g., `ganblr_plus`).

---

## 🗂 Step 2: Add Your Model

Use the scaffold tool to generate the boilerplate automatically:

```bash
python scaffold.py init-model <model_name> dep1 dep2
```

This creates `katabatic/models/<model_name>/` with `__init__.py`, `models.py`, and `utils.py` pre-filled, and registers the model in `katabatic/models/registry.py`.

Then:

1. Implement `train()`, `sample()`, and `evaluate()` in `models.py`
2. Add your model's dependencies to the root `pyproject.toml`:
   - Under `[tool.poetry.dependencies]` as optional
   - Under `[tool.poetry.extras]` with the model name as the key
3. Your model class extends the `Model` base class defined in:

   ```python
   from katabatic.models.base_model import Model
   ```

   This ensures consistency across all models and compatibility with the evaluation pipeline.

---

## ✅ Step 3: Finish and Push

Once development is complete and tested:

```bash
git add .
git commit -m "Add <model_name> model"
git push origin feature/<model_name>
```

---

## 🔁 Step 4: Open a Pull Request

Create a **Pull Request (PR)** from your feature branch into the `development` branch. Make sure to:

- Include a summary of your model
- Mention any new dependencies
- Add evaluation results if applicable (see `benchmarks/results/` for examples)

---

## 📁 Optional: Data and Results

- Place **synthetic data outputs** under:
  ```
  benchmarks/synthetic/<dataset>/<model_name>/
  ```
- Evaluation results go into:
  ```
  benchmarks/results/<dataset>/<model_name>/
  ```

---

## 🤝 Thanks for Contributing!

Keep contributions modular and follow the code style used in the repo for smooth integration.

---

## 🧹 Code Formatting

Before pushing your changes, ensure your Python code is formatted using [autopep8](https://pypi.org/project/autopep8/). You can do this by running:

```bash
autopep8 --in-place --recursive .
```

This helps maintain consistent code style across the project and makes code reviews smoother.
