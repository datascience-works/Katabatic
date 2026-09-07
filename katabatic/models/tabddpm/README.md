## TabDDPM: Modelling Tabular Data with Diffusion Models

Production-level implementation of TabDDPM for the Katabatic framework.

## Paper
https://github.com/rotot0/tab-ddpm 
https://arxiv.org/abs/2209.15421 
TabDDPM: Modelling Tabular Data with Diffusion Models

Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, and Artem Babenko

ICML 2023

arXiv:2209.15421

## Overview

TabDDPM is a diffusion-based generative model designed specifically for tabular data. It supports mixed data types by using Gaussian diffusion for numerical features and multinomial diffusion for categorical and binary features. The reverse diffusion process is modeled using an MLP-based denoising network.

## Installation

poetry install --extras tabddpm

## Usage

from katabatic.models.tabddpm.models import Tabddpm

from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

pipeline = TrainTestSplitPipeline(model=Tabddpm())

pipeline._evaluations = []

## Dataset Compatibility

The TabDDPM implementation has been validated with the main datasets currently available in the Katabatic project, including Car, Nursery, Adult, MAGIC, and Shuttle.

Car contains fully categorical or discrete data and is compatible with the model. Nursery also contains fully categorical or discrete data and is compatible. Adult contains a mixture of categorical and numerical features and is strongly suited to TabDDPM because the model is designed to handle mixed tabular data. MAGIC mainly contains numerical features, while Shuttle contains numerical features, and both are also compatible with the model.

## Best Matching Dataset

Among the datasets currently available in the Katabatic project, Adult is the best matching dataset for the original TabDDPM research paper.

The reason is that Adult is directly included in the benchmark datasets used in the original TabDDPM research paper. In the paper, the Adult dataset contains 26,048 training samples, 6,513 validation samples, and 16,281 test samples. It contains 6 numerical features and 8 categorical features and is used for a binary classification task.

Therefore, Adult provides a direct research-paper-aligned dataset for the current TabDDPM benchmarking work. This is different from MedGAN, where the original medical datasets are not available in the Katabatic project and a structurally similar dataset must be selected instead.

## Current Update

The TabDDPM implementation has been validated with the current Katabatic training and synthetic data generation pipeline. Initial baseline testing has been completed on available datasets including Car and Nursery.

The Nursery experiment confirmed that TabDDPM can train successfully, generate synthetic records, and run through the Katabatic synthetic data evaluation pipeline.

For Sprint 2, Adult has been selected as the primary dataset for research-paper-aligned TabDDPM benchmarking because the same dataset is used in the original TabDDPM research paper. Adult also contains both categorical and numerical features, which closely matches the type of heterogeneous tabular data TabDDPM was designed to model.

The next stage of the work is to run TabDDPM on Adult, record the model configuration and training results, evaluate the generated synthetic data, and compare the results with the research paper and the Katabatic benchmarking framework.

## Key Features

TabDDPM supports both numerical and categorical features. It uses Gaussian diffusion for numerical variables and multinomial diffusion for categorical variables. It is suitable for mixed-type tabular datasets and supports reproducible synthetic sampling. The implementation is integrated with the Katabatic training pipeline and can be evaluated using the Katabatic synthetic data evaluation framework.

## Evaluation

The TabDDPM implementation can be evaluated using the Katabatic Synthetic Evaluation Pipeline. The current project evaluates generated data using fidelity, utility, diversity, privacy, consistency, and stability.

These are Katabatic project-level benchmarking metrics and are not identical to all evaluation measures used in the original TabDDPM research paper.

## Citation

@inproceedings{kotelnikov2023tabddpm,
title={TabDDPM: Modelling Tabular Data with Diffusion Models},
author={Kotelnikov, Akim and Baranchuk, Dmitry and Rubachev, Ivan and Babenko, Artem},
booktitle={Proceedings of the 40th International Conference on Machine Learning},
year={2023}
}