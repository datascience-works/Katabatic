from abc import ABC, abstractmethod

import pandas as pd


class Evaluation(ABC):
    """
    Abstract base class for all evaluation dimensions.

    Each evaluator receives the real and synthetic datasets as DataFrames
    and returns a structured result dict from evaluate().
    """

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame = None):
        self.real_data = real_data
        self.synthetic_data = synthetic_data

    @abstractmethod
    def evaluate(self) -> dict:
        """
        Run the evaluation and return a structured result dict.
        Must include a normalized score key
        """
        raise NotImplementedError("Subclasses must implement evaluate().")
