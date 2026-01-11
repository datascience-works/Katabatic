
class CrossValidationPipeline:
    """
    A class to handle the cross-validation pipeline for model evaluation.
    This class is responsible for managing the cross-validation process,
    including splitting the dataset, training models, and evaluating their performance.
    """

    def __init__(self, model, dataset, n_splits=5):
        self.model = model
        self.dataset = dataset
        self.n_splits = n_splits

    def run(self):
        """
        Execute the cross-validation process.
        """
        ...

    def evaluate(self):
        """
        Evaluate the model's performance using cross-validation metrics.
        """
        ...



# first half train, second half test then second half train, first half test