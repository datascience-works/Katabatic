# 🧬 GReaT Synthetic Data Generation and Benchmarking on Sick Dataset

This project explores the use of the **GReaT** (Generative ReaT) model for generating synthetic healthcare data and benchmarking it against real data.  
The focus is on the **Sick dataset** (`dataset_38_sick.arff`), with experiments conducted in **classification tasks** to evaluate the effectiveness of synthetic data in comparison to real-world samples.

---

## 📂 Project Structure
- `dataset_38_sick.arff` → Original ARFF dataset  
- `real_sick.csv` → Converted real dataset  
- `synthetic_sick.csv` → Generated synthetic dataset  
- `sick_models.py` → Core model training and benchmarking  
- `sick_utils.py` → Helper functions for preprocessing  
- `benchmarks_sick.py` → Benchmark experiments (real vs synthetic training)  
- `sick_pyproject.toml` → Project dependencies for reproducibility  
- `README.md` → Project documentation (this file)  

---

## ⚙️ Installation

Run the following inside your Colab or local environment:

```bash
pip install be-great
pip install torch scikit-learn pandas numpy scipy transformers

🚀 Usage

Load Dataset

Convert ARFF to DataFrame

Clean and save to real_sick.csv

Train GReaT Model

Backbone: distilgpt2

Configured with early stopping, evaluation, and logging

Trained for 3–5 epochs (for quick experimentation)

Generate Synthetic Data

Sampled with temperature and guided sampling

Saved as synthetic_sick.csv

Benchmark Models

Classification: Logistic Regression, Decision Tree, Random Forest

Regression (optional): Linear Regression, Decision Tree, Random Forest

Comparison setups:

Train on Real → Test on Real

Train on Synthetic → Test on Real

📊 Results
✅ Dataset Shapes

Real dataset: (3772, 30)

Synthetic dataset: (50, 30)

Dropped all-NaN columns: ['TBG']

⚖️ Class Distribution

Real Dataset: Imbalanced (majority negative)

Synthetic Dataset: Multiple classes generated, sometimes spurious

📝 Benchmark Outcomes
Model	Setup	Accuracy	Precision	Recall	F1
Logistic Regression	Train Real → Test Real	0.96	0.96	0.96	0.96
Logistic Regression	Train Synthetic → Test Real	0.75	0.90	0.75	0.82
Decision Tree	Train Real → Test Real	0.99	0.99	0.99	0.99
Decision Tree	Train Synthetic → Test Real	0.91	0.88	0.91	0.90
Random Forest	Train Real → Test Real	0.99	0.99	0.99	0.99
Random Forest	Train Synthetic → Test Real	0.94	0.88	0.94	0.91
🏆 Contributions

As part of this project, I (Arpit) contributed across the entire lifecycle of the experiment — from dataset preparation to model training, benchmarking, and reporting.

1. Dataset Preparation

Data Acquisition: Used dataset_38_sick.arff as the primary real dataset.

Format Conversion: Converted ARFF → CSV (real_sick.csv) for ML workflows.

Data Cleaning:

Decoded byte-encoded categorical values.

Handled missing values (NaN).

Dropped fully empty columns like TBG.

Backup Handling: Ensured clean dataset was always saved for reproducibility.

2. Model Training with GReaT

Configured GReaT model with distilgpt2 backbone.

Set training parameters: batch size, epochs, CPU/GPU fallback.

Integrated early stopping and logging.

Ran multiple training rounds and monitored loss curves.

3. Synthetic Data Generation

Generated synthetic datasets (synthetic_sick.csv).

Experimented with temperature and guided sampling.

Implemented CPU fallback for sampling reliability.

Validated schema, alignment, and class distribution vs real dataset.

4. Benchmarking and Model Evaluation

Designed experiments:

Train on Real → Test on Real

Train on Synthetic → Test on Real

Implemented multiple ML models:

Classification: Logistic Regression, Decision Tree, Random Forest

Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor

Built pipelines: imputation, encoding, preprocessing.

Measured metrics: Accuracy, Precision, Recall, F1, MSE, RMSE, MAE, R².

5. Result Analysis

Observed imbalance in real dataset.

Found synthetic dataset introduced spurious classes.

Showed Random Forest had best performance (~99% on real, ~94% synthetic).

Debugged errors: mismatched labels, imputation issues, NaN handling.

6. Documentation & Reporting

Authored this README.md.

Documented full workflow, results, and limitations.

Collected screenshots, logs, and metrics for reporting.

Suggested future improvements.

🔮 Future Directions

Train longer (more epochs) for better convergence.

Balance class labels in synthetic data.

Explore larger LLMs (GPT-Neo, GPT-J) as backbones.

Add Gradient Boosting, XGBoost benchmarks.

Extend regression evaluations.

👨‍💻 Author

Arpit (s224752512)
Master’s in IT (Professional)