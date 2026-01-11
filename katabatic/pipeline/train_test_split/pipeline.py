from katabatic.pipeline.base_pipeline import Pipeline
from katabatic.models.base_model import Model
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.utils.split_dataset import split_dataset


class TrainTestSplitPipeline(Pipeline):
    """
    Version 1 of the pipeline.
    This is a placeholder for future enhancements.
    """
    _evaluations = [TSTREvaluation]

    def __init__(self, model: Model, evaluations=None, override_evaluations=False):
        super().__init__(model)

        if evaluations and override_evaluations:
            self._evaluations = evaluations
        elif evaluations:
            self._evaluations.extend(evaluations)

    def run(self, *args, **kwargs):
        """
        Run the train test split pipeline with the given arguments.
        """
        current_model = self.model()

        input_csv = kwargs.pop('input_csv', None)
        output_dir = kwargs.pop('output_dir', None)

        if not input_csv or not output_dir:
            raise ValueError(
                "Both 'input_csv' and 'output_dir' must be provided.")

        split_dataset(input_csv, output_dir, *args, **kwargs)

        # Train the model (may consume extra kwargs like 'config')
        current_model.train(output_dir, *args, **kwargs)

        # Filter kwargs for evaluations to avoid unexpected params (e.g., 'config')
        eval_kwargs = dict(kwargs)
        eval_kwargs.pop('config', None)

        for evaluation in self._evaluations:
            eval_instance = evaluation(*args, **eval_kwargs)
            eval_instance.evaluate()

        # Implement the specific logic for version 1 pipeline here
        return "Train test split pipeline executed successfully."

    def __repr__(self):
        return f"TrainTestSplitPipeline(name={self.model})"
