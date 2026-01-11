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

Inside the `katabatic/models/` directory:

1. Create a new folder for your model:

   ```
   katabatic/models/<model_name>/
   ```

2. Within that folder, follow the format used in existing models (like `ganblr` or `great`). Typically, this includes:

   - `__init__.py`
   - `models.py`
   - `utils.py` _(if needed)_
   - `pyproject.toml` and `poetry.lock` _(if dependencies are isolated)_

3. Your model class should **extend** the `Model` base class defined in:

   ```python
   from katabatic.models.base_model import Model
   ```

   This ensures consistency across all models and compatibility with the evaluation and pipeline systems.

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
- Add evaluation results if applicable (see `Results/` for examples)

---

## 📁 Optional: Data and Results

- Place **synthetic data outputs** under:
  ```
  synthetic/<dataset>/<model_name>/
  ```
- Evaluation results go into:
  ```
  Results/<dataset>/<model_name>_tstr.csv
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
