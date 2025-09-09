## The Elusive Pursuit of Reproducing PATE-GAN: Benchmarking, Auditing, Debugging

This repository contains the source code for the paper The Elusive Pursuit of Reproducing PATE-GAN: Benchmarking, Auditing, Debugging by G. Ganev, M.S.M.S. Annamalai, E. De Cristofaro, [TMLR 2025](https://openreview.net/forum?id=wcxrJcJ7vq)


## Install

The experiments require Python 3.10.
All necessary dependencies are listed in `requirements.txt`.
Since there are conflicts between some libraries, i.e., `synthcity` and `smartnoise-synth`, it is recommended to install the dependencies manually.

## Usage
The `example.ipynb` shows how generate synthetic datasets based on the original Pate-Gans architecture. The pipeline does not support preprocessing, so NaN values and other stuff need to be cleaned before fitting.
