"""
GANBLR: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis

GANBLR models created by Tulip Lab at Deakin University for tabular data generation, which can sample fully artificial data from real data.
A novel tabular data generation model– Generative Adversarial Network modelling inspired from Naive Bayes 
and Logistic Regression’s relationship (GANBLR).

Paper: https://www.computer.org/csdl/proceedings-article/icdm/2021/239800a916/1Aqx3a4iqsw
"""

from katabatic.models.ganblr.models import GANBLR

__all__ = ['GANBLR']
__version__ = '1.0.0'
