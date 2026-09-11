import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from katabatic.models.base_model import Model

from .utils import build_conditional_tables, encode_features, sample_from_tables


class NaiveBayesSynth(Model):
    """
    A simple categorical Naive-Bayes-style synthetic data generator.

    Learns P(class) and, for every feature, P(feature=category | class) from
    the real training data (with Laplace smoothing), then generates new rows
    by sampling a class from the prior and sampling each feature
    independently from its class-conditional distribution.

    This is intentionally simple -- it assumes features are conditionally
    independent given the class (the "naive" assumption) -- and is meant as
    a light-weight baseline generator, not a competitor to GANBLR-style
    models that capture feature dependencies.
    """

    ARTIFACT_STATE_FILES = "naive_bayes_synth_model.pkl"

    def __init__(self) -> None:
        super().__init__()
        self.check_dependencies()
        self._feature_encoder = None
        self._label_encoder = LabelEncoder()
        self._class_priors = None
        self._conditional_tables = None
        self._feature_names = None
        self._target_name = None
        self.smoothing = 1.0

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["numpy", "pandas", "sklearn"]

    def fit(self, x, y, smoothing: float = 1.0, verbose: int = 1):
        """
        Fit the model to the given data.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features)
            Dataset to fit the model. The data should be discrete/categorical.
        y : array_like of shape (n_samples,)
            Label of the dataset.
        smoothing : float, default=1.0
            Laplace smoothing constant applied to every conditional probability
            table, so unseen (feature, class) combinations don't get zero
            probability.
        verbose : int, default=1
            Whether to print a short fit summary. Use 0 for silence.

        Returns
        -------
        self : object
            Fitted model.
        """
        self.smoothing = smoothing
        self._feature_names = (
            list(x.columns) if hasattr(x, "columns") else [f"feature_{i}" for i in range(np.asarray(x).shape[1])]
        )
        self._target_name = y.name if hasattr(y, "name") and y.name else "target"

        x_enc, self._feature_encoder = encode_features(pd.DataFrame(x, columns=self._feature_names))
        y_enc = self._label_encoder.fit_transform(y).astype(int)
        num_classes = len(self._label_encoder.classes_)

        class_counts = np.bincount(y_enc, minlength=num_classes).astype(float)
        self._class_priors = class_counts / class_counts.sum()
        self._conditional_tables = build_conditional_tables(
            x_enc, y_enc, num_classes, smoothing=self.smoothing
        )

        if verbose:
            print(
                f"Fitted NaiveBayesSynth on {x_enc.shape[0]} rows, "
                f"{x_enc.shape[1]} features, {num_classes} classes."
            )
        return self

    def sample(self, size=None, seed=None) -> pd.DataFrame:
        """
        Generate synthetic data.

        Parameters
        ----------
        size : int or None
            Number of rows to generate. Defaults to the training set size.
        seed : int or None, default=None
            Random seed for reproducible sampling.

        Returns
        -------
        synthetic_samples : pd.DataFrame
            Generated synthetic data, with the same column names used at fit time.
        """
        if self._conditional_tables is None:
            raise RuntimeError("Model must be fit before sampling.")
        rng = np.random.default_rng(seed)
        n = size if size is not None else len(self._class_priors)
        x_synth_enc, y_synth_enc = sample_from_tables(
            self._class_priors, self._conditional_tables, n, rng
        )
        x_synth = self._feature_encoder.inverse_transform(x_synth_enc)
        y_synth = self._label_encoder.inverse_transform(y_synth_enc).reshape(-1, 1)

        columns = self._feature_names + [self._target_name]
        return pd.DataFrame(np.hstack([x_synth, y_synth]), columns=columns)

    def evaluate(self, x, y, model="lr") -> float:
        """
        Perform a TSTR (Train on Synthetic, Test on Real) evaluation.

        Parameters
        ----------
        x, y : array_like
            Real held-out test dataset.
        model : str or object
            One of ['lr', 'mlp', 'rf'], or an object with sklearn-style
            `fit`/`predict` methods.

        Returns
        -------
        accuracy_score : float
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        models = dict(lr=LogisticRegression, rf=RandomForestClassifier, mlp=MLPClassifier)
        if model in models:
            eval_model = models[model]()
        elif hasattr(model, "fit") and hasattr(model, "predict"):
            eval_model = model
        else:
            raise ValueError(
                "Invalid `model` argument. Use one of ['lr', 'mlp', 'rf'], "
                "or an object with sklearn-style fit/predict methods."
            )

        synth = self.sample(size=max(len(x), 500))
        x_synth, y_synth = synth[self._feature_names], synth[self._target_name]

        x_synth_enc = self._feature_encoder.transform(x_synth)
        y_synth_enc = self._label_encoder.transform(y_synth)
        x_test_enc = self._feature_encoder.transform(pd.DataFrame(x, columns=self._feature_names))
        y_test_enc = self._label_encoder.transform(y)

        pipeline = Pipeline(
            [
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("model", eval_model),
            ]
        )
        pipeline.fit(x_synth_enc, y_synth_enc)
        pred = pipeline.predict(x_test_enc)
        return accuracy_score(y_test_enc, pred)

    def train(self, dataset, size_category="small", *args, **kwargs):
        """
        Pipeline entrypoint: loads x_train.csv/y_train.csv for `dataset`,
        fits the model, generates synthetic data, and writes it to
        synthetic/<dataset>/naive_bayes_synth/ (or `synthetic_dir` if given).
        """
        model_name = "naive_bayes_synth"
        data_dir = dataset

        explicit_synth_dir = kwargs.get("synthetic_dir")
        save_dir = explicit_synth_dir or os.path.join("synthetic", dataset, model_name)
        os.makedirs(save_dir, exist_ok=True)

        x_train_path = os.path.join(data_dir, "x_train.csv")
        y_train_path = os.path.join(data_dir, "y_train.csv")

        x = pd.read_csv(x_train_path)
        y_df = pd.read_csv(y_train_path)
        y = y_df.iloc[:, 0]

        smoothing = kwargs.get("smoothing", 1.0)
        self.fit(x, y, smoothing=smoothing, verbose=kwargs.get("verbose", 1))

        df_synth = self.sample(size=len(x), seed=kwargs.get("seed"))
        x_synth = df_synth.iloc[:, :-1]
        y_synth = df_synth.iloc[:, -1]

        x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"), index=False, header=True)
        print(f"Synthetic data saved to: {save_dir}")

        artifact_state_dir = kwargs.get("artifact_state_dir")
        if artifact_state_dir:
            os.makedirs(artifact_state_dir, exist_ok=True)
            try:
                import pickle

                with open(os.path.join(artifact_state_dir, self.ARTIFACT_STATE_FILES), "wb") as f:
                    pickle.dump(self, f)
            except Exception as e:
                print(f"Failed to dump pickle file: {e}")