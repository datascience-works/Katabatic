from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train():
        ...

    @abstractmethod
    def evaluate():
        ...

    @abstractmethod
    def sample():
        ...
