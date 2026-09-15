from katabatic.models.base_model import Model


class Pipeline:
    """
    Base class for all pipelines.
    """

    def __init__(self, model: Model):
        self.model = model

    def run(self, *args, **kwargs):
        """
        Run the pipeline with the given arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __repr__(self):
        return f"Pipeline(model={self.model.__str__()})"
