# Katabatic — Project Architecture

## Overview

Katabatic is a library of tabular data generative models with a 6-dimension evaluation pipeline. Every model follows the same abstract interface, and any generated synthetic dataset can be scored through the same pipeline regardless of which model produced it.

---

## Full Architecture

```mermaid
flowchart TD
    %% ── Data Preparation ──────────────────────────────────────
    A[(Raw Dataset\nCSV)] -->|preprocess_tabular| B[Preprocessed Data\ncleaned, original col names + types preserved]
    B -->|split_dataset\nstratified 80/20| C[Train Split] & D[Test Split]

    %% ── Model Training ────────────────────────────────────────
    C -->|model.train| E[Generative Model]

    subgraph models [katabatic/models/]
        direction LR
        M1[CTGAN]
        M2[CoDi]
        M3[TabDDPM]
        M4[GANBLR]
        M5[GReaT]
        M6[Tabsyn]
        M7[MedGAN]
        M8[PATEGAN]
    end

    E --- models

    %% ── Synthetic Data Generation ─────────────────────────────
    E -->|model.sample| F[(Synthetic\nDataFrame)]

    %% ── Evaluation Pipeline ───────────────────────────────────
    C & D & F -->|train + test + synthetic| G

    subgraph pipeline [katabatic/pipeline/evaluation_pipeline.py]
        G[SyntheticEvaluationPipeline]

        G --> FID[Fidelity\nJSD · Wasserstein · Correlation]
        G --> UTL[Utility\nTSTR vs TRTR · 5 classifiers]
        G --> DIV[Diversity\nCategory · Bin · Gower Coverage]
        G --> PRV[Privacy\nNNDR · Exact · Near Duplicates]
        G --> CON[Consistency\nDiscriminator · Constraints · Feature Importance]
        G --> STB[Stability\nMulti-run Variance · seeds 0–4]
    end

    %% ── Report ────────────────────────────────────────────────
    FID & UTL & DIV & PRV & CON & STB --> R

    subgraph report [katabatic/evaluate/report/composite.py]
        R[EvaluationReport]
        R --> W["Weighted Composite Score [0–1]\nUtility 35% · Fidelity 25% · Privacy 15%\nDiversity 10% · Consistency 10% · Stability 5%"]
    end

    W --> OUT1[JSON Report]
    W --> OUT2[CSV Summary]
    W --> OUT3[Console Output]

    %% ── Runner helper (benchmarks only) ───────────────────────
    subgraph runner [benchmarks/runner.py]
        RC[RunConfig\ndataset · model · columns · constraints]
        RC --> PPS[preprocess_and_split]
        RC --> SS[save_synthetic]
        RC --> EV[evaluate]
    end
```

---

## Abstract Base Classes

```mermaid
classDiagram
    class Model {
        <<abstract>>
        +train(dataset_dir, **kwargs) Model
        +sample(n_samples, **kwargs) DataFrame
        +evaluate(**kwargs) float
        +check_dependencies() bool
    }

    class Evaluation {
        <<abstract>>
        +real_data: DataFrame
        +synthetic_data: DataFrame
        +evaluate() dict
    }

    class Pipeline {
        <<abstract>>
        +run(*args, **kwargs)
    }

    Model <|-- CTGANModel
    Model <|-- CODI
    Model <|-- Tabddpm
    Model <|-- GANBLR
    Model <|-- GReaT
    Model <|-- Tabsyn
    Model <|-- MedGANSynthesizer
    Model <|-- PATEGANSynthesizer

    Evaluation <|-- FidelityEvaluation
    Evaluation <|-- UtilityEvaluation
    Evaluation <|-- DiversityEvaluation
    Evaluation <|-- PrivacyEvaluation
    Evaluation <|-- ConsistencyEvaluation
    Evaluation <|-- StabilityEvaluation

    Pipeline <|-- SyntheticEvaluationPipeline

    SyntheticEvaluationPipeline --> Evaluation : orchestrates
    SyntheticEvaluationPipeline --> EvaluationReport : returns
```

---

## Data Flow

```mermaid
flowchart LR
    A[raw CSV] --> B[preprocess_tabular\ncleaned CSV]
    B --> C[train_full.csv] & T[test_full.csv]
    C --> D[model.train]
    D --> E[model.sample]
    E --> F[synthetic DataFrame]
    C & T & F --> G[SyntheticEvaluationPipeline]
    G --> H[EvaluationReport\ncomposite_score\ndimension_scores]
```

---

## Directory Structure

```
katabatic/
├── models/
│   ├── base_model.py          # Abstract Model base class
│   ├── registry.py            # Dynamic model loader
│   ├── ctgan/                 # CTGAN implementation
│   ├── codi/                  # CoDi implementation
│   ├── tabddpm/               # TabDDPM implementation
│   ├── ganblr/                # GANBLR implementation
│   ├── great/                 # GReaT implementation
│   ├── tabsyn/                # Tabsyn implementation
│   ├── medgan/                # MedGAN implementation
│   └── pategan/               # PATEGAN implementation
│
├── evaluate/
│   ├── base_evaluation.py     # Abstract Evaluation base class
│   ├── fidelity/              # JSD + Wasserstein + Correlation
│   ├── utility/               # TSTR vs TRTR across 5 classifiers
│   ├── diversity/             # Category + Bin + Gower coverage
│   ├── privacy/               # NNDR + duplicate detection
│   ├── consistency/           # Discriminator + constraints + feature importance
│   ├── stability/             # Multi-run variance
│   └── report/                # EvaluationReport + composite scoring
│
├── pipeline/
│   ├── base_pipeline.py       # Abstract Pipeline base class
│   └── evaluation_pipeline.py # SyntheticEvaluationPipeline (main)
│
└── utils/
    ├── column_types.py        # Categorical/continuous auto-detection
    ├── split_dataset.py       # Stratified train/test split
    └── preprocess.py          # preprocess_tabular + data cleaning

benchmarks/
├── runner.py                  # RunConfig + shared pipeline helpers
└── examples/                  # reference run scripts — copy and adapt for your model
    ├── run_ctgan_adult.py
    ├── run_codi_adult.py
    ├── run_tabddpm_adult.py
    └── run_ctgan_bank_marketing.py
datasets/
├── adult.csv
└── bank_marketing.csv
```
