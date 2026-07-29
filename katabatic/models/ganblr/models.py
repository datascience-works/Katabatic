from __future__ import annotations

import logging
import os
import random
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from katabatic.models.base_model import Model

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef

from .kdb import _add_uniform
from .utils import DataUtils, _ensure_tf, elr_loss, get_lr, sample, softmax_weight


@contextmanager
def _ganblr_quiet_logs():
    """Reduce pgmpy / Keras noise during sampling and small-model builds."""
    pg = logging.getLogger("pgmpy")
    prev_pg = pg.level
    pg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            yield
    finally:
        pg.setLevel(prev_pg)


def _epoch_progress(epochs: int, verbose: int):
    """
    Returns (iterable, tqdm_bar_or_None). When tqdm is available, iterable is the bar
    (supports set_postfix); otherwise range(epochs).
    """
    if verbose <= 0:
        return range(epochs), None
    try:
        from tqdm.auto import tqdm

        bar = tqdm(
            range(epochs),
            desc="GANBLR",
            unit="epoch",
            leave=True,
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )
        return bar, bar
    except ImportError:
        return range(epochs), None


class GANBLR(Model):
    """
    The GANBLR Model.
    """

    #: Filenames written under artifact ``state/`` (used to detect reloadable train runs).
    ARTIFACT_STATE_FILES = ("ganblr_model.pkl",)

    def __init__(self) -> None:
        super().__init__()
        self.check_dependencies()  # Check dependencies on initialization
        self._d = None
        self.__gen_weights = None
        self.batch_size = None
        self.epochs = None
        self.k = None
        self.constraints = None
        self._ordinal_encoder = OrdinalEncoder(
            dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        self._label_encoder = LabelEncoder()

    def check_dependencies(self) -> bool:
        """
        Verify optional deps without importing TensorFlow at construction time.

        TF is loaded lazily on first ``fit()`` via ``_ensure_tf()`` so notebooks and
        pipelines can build ``GANBLR()`` without a long silent import.
        """
        required_deps = [
            d for d in self.get_required_dependencies() if d != "tensorflow"
        ]
        missing_deps: list[str] = []
        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            raise ImportError(
                f"Missing required dependencies for {self.__class__.__name__}: {missing_deps}. "
                f"Install with: pip install katabatic[{self.__class__.__name__.lower()}]"
            )

        return True

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return a list of required dependencies for this model."""
        return ['tensorflow', 'pgmpy', 'sklearn', 'scipy']

    def fit(self, x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=1):
        '''
        Fit the model to the given data.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features)
            Dataset to fit the model. The data should be discrete.

        y : array_like of shape (n_samples,)
            Label of the dataset.

        k : int, default=0
            Parameter k of ganblr model. Must be greater than 0. No more than 2 is Suggested.

        batch_size : int, default=32
            Size of the batch to feed the model at each step.

        epochs : int, default=0
            Number of epochs to use during training.

        warmup_epochs : int, default=1
            Number of epochs to use in warmup phase. Defaults to :attr:`1`.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        Returns
        -------
        self : object
            Fitted model.
        '''
        epsilon = 1e-10

        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        _ensure_tf(announce=verbose > 0)
        x = self._ordinal_encoder.fit_transform(x)
        y = self._label_encoder.fit_transform(y).astype(int)
        d = DataUtils(x, y)
        self._d = d
        self.k = k
        self.batch_size = batch_size
        if verbose:
            print("GANBLR: warmup (encoder + generator)…", flush=True)
        _ = self._warmup_run(warmup_epochs, verbose=0)
        if verbose:
            print("GANBLR: warmup done.", flush=True)

        discriminator_label = np.hstack(
            [np.ones(d.data_size), np.zeros(d.data_size)])

        epoch_iter, pbar = _epoch_progress(epochs, verbose)
        with _ganblr_quiet_logs():
            syn_data = self._sample(verbose=0)
            for i in epoch_iter:
                discriminator_input = np.vstack([x, syn_data[:, :-1]])
                disc_input, disc_label = sample(
                    discriminator_input, discriminator_label, frac=0.8)
                disc = self._discrim()
                d_history = disc.fit(
                    disc_input, disc_label, batch_size=batch_size, epochs=1, verbose=0).history
                prob_fake = disc.predict(x, verbose=0)
                ls = np.mean(-np.log(np.clip(1 - prob_fake, epsilon, 1)))
                g_history = self._run_generator(loss=ls).history
                syn_data = self._sample(verbose=0)

                if pbar is not None:
                    pbar.set_postfix(
                        G_loss=f"{g_history['loss'][0]:.3f}",
                        D_loss=f"{d_history['loss'][0]:.3f}",
                        G_acc=f"{g_history['accuracy'][0]:.3f}",
                        D_acc=f"{d_history['accuracy'][0]:.3f}",
                    )
                elif verbose:
                    print(
                        f"Epoch {i + 1}/{epochs}: G_loss = {g_history['loss'][0]:.6f}, "
                        f"G_accuracy = {g_history['accuracy'][0]:.6f}, D_loss = {d_history['loss'][0]:.6f}, "
                        f"D_accuracy = {d_history['accuracy'][0]:.6f}",
                        flush=True,
                    )
        if pbar is not None:
            pbar.close()
        self.is_fitted = True
        return self

    def evaluate(self, x, y, model='lr') -> float:
        """
        In-memory TSTR-style check: train a small sklearn pipeline on fresh
        synthetic samples and score on the provided real ``(x, y)`` test set.

        For **canonical** utility metrics (multiple classifiers, CSV-based split
        aligned with training), use :class:`katabatic.pipeline.train_test_split.pipeline.TrainTestSplitPipeline`
        with :class:`katabatic.evaluate.tstr.evaluation.TSTREvaluation`.

        Parameters
        ----------
        x, y : array_like
            Test dataset.

        model : str or object
            The model used for evaluate. Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.

        Return:
        --------
        accuracy_score : float.

        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        eval_model = None
        models = dict(
            lr=LogisticRegression,
            rf=RandomForestClassifier,
            mlp=MLPClassifier
        )
        if model in models:
            eval_model = models[model]()
        elif hasattr(model, 'fit') and hasattr(model, 'predict'):
            eval_model = model
        else:
            raise Exception(
                "Invalid Arugument `model`, Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.")

        synthetic_data = self._sample()
        synthetic_x, synthetic_y = synthetic_data[:,
                                                  :-1], synthetic_data[:, -1]
        x_test = self._ordinal_encoder.transform(x)
        y_test = self._label_encoder.transform(y)

        categories = self._d.get_categories()
        pipline = Pipeline([('encoder', OneHotEncoder(
            categories=categories, handle_unknown='ignore')), ('model',  eval_model)])
        pipline.fit(synthetic_x, synthetic_y)
        pred = pipline.predict(x_test)
        return accuracy_score(y_test, pred)

    def sample(self, size=None, verbose=1) -> np.ndarray:
        """
        Generate synthetic data.     

        Parameters
        ----------
        size : int or None
            Size of the data to be generated. set to `None` to make the size equal to the size of the training set.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        Return:
        -----------------
        synthetic_samples : np.ndarray
            Generated synthetic data.
        """
        ordinal_data = self._sample(size, verbose)
        origin_x = self._ordinal_encoder.inverse_transform(
            ordinal_data[:, :-1])
        origin_y = self._label_encoder.inverse_transform(
            ordinal_data[:, -1]).reshape(-1, 1)
        return np.hstack([origin_x, origin_y])

    def _sample(self, size=None, verbose=1) -> np.ndarray:
        """
        Generate synthetic data in ordinal encoding format
        """
        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        # basic varibles
        d = self._d
        feature_cards = np.array(d.feature_uniques)
        # ensure sum of each constraint group equals to 1, then re concat the probs
        _idxs = np.cumsum([0] + d._kdbe.constraints_.tolist())
        constraint_idxs = [(_idxs[i], _idxs[i+1]) for i in range(len(_idxs)-1)]

        probs = np.exp(self.__gen_weights[0])
        cpd_probs = [probs[start:end, :] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p/p.sum(axis=0) for p in cpd_probs])

        # assign the probs to the full cpd tables
        idxs = np.cumsum([0] + d._kdbe.high_order_feature_uniques_)
        feature_idxs = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
        have_value_idxs = d._kdbe.have_value_idxs_
        full_cpd_probs = []
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            # (n_high_order_feature_uniques, n_classes)
            cpd_prob_ = cpd_probs[start:end, :]
            # (n_all_combination) Note: the order is (*parent, variable)
            have_value_ravel = have_value.ravel()
            # (n_classes * n_all_combination)
            have_value_ravel_repeat = np.hstack(
                [have_value_ravel] * d.num_classes)
            # (n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
            full_cpd_prob_ravel = np.zeros_like(
                have_value_ravel_repeat, dtype=float)
            full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
            # (n_classes * n_parent_combinations, n_variable_unique)
            full_cpd_prob = full_cpd_prob_ravel.reshape(
                -1, have_value.shape[-1]).T
            full_cpd_prob = _add_uniform(full_cpd_prob, noise=0)
            full_cpd_probs.append(full_cpd_prob)

        # prepare node and edge names
        node_names = [str(i) for i in range(d.num_features + 1)]
        edge_names = [(str(i), str(j)) for i, j in d._kdbe.edges_]
        y_name = node_names[-1]

        # create TabularCPD objects
        evidences = d._kdbe.dependencies_
        feature_cpds = [
            TabularCPD(str(name), feature_cards[name], table,
                       evidence=[y_name, *[str(e) for e in evidences]],
                       evidence_card=[d.num_classes, *feature_cards[evidences].tolist()])
            for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
        ]
        y_probs = (d.class_counts/d.data_size).reshape(-1, 1)
        y_cpd = TabularCPD(y_name, d.num_classes, y_probs)

        # create kDB model, then sample the data
        model = DiscreteBayesianNetwork(edge_names)
        model.add_cpds(y_cpd, *feature_cpds)
        sample_size = d.data_size if size is None else size
        result = BayesianModelSampling(model).forward_sample(
            size=sample_size, show_progress=verbose > 0)
        sorted_result = result[node_names].values

        return sorted_result

    def _warmup_run(self, epochs, verbose=None):
        tf = _ensure_tf()
        d = self._d
        tf.keras.backend.clear_session()
        ohex = d.get_kdbe_x(self.k)
        self.constraints = softmax_weight(d.constraint_positions)
        elr = get_lr(ohex.shape[1], d.num_classes, self.constraints)
        history = elr.fit(ohex, d.y, batch_size=self.batch_size,
                          epochs=epochs, verbose=verbose)
        self.__gen_weights = elr.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _run_generator(self, loss):
        tf = _ensure_tf()
        d = self._d
        ohex = d.get_kdbe_x(self.k)
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            d.num_classes, input_dim=ohex.shape[1], activation='softmax', kernel_constraint=self.constraints))
        model.compile(loss=elr_loss(loss), optimizer='adam',
                      metrics=['accuracy'])
        model.set_weights(self.__gen_weights)
        history = model.fit(
            ohex, d.y, batch_size=self.batch_size, epochs=1, verbose=0)
        self.__gen_weights = model.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _discrim(self):
        tf = _ensure_tf()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            1, input_dim=self._d.num_features, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def _save_artifact_state(self, state_dir: str) -> None:
        import pickle

        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, self.ARTIFACT_STATE_FILES[0])
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[GANBLR] Saved fitted model state to: {path}")

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> GANBLR:
        import pickle

        import joblib

        rel = f"{ref.state_relpath}/{cls.ARTIFACT_STATE_FILES[0]}"
        path = store.open_path(rel)
        if not path.is_file():
            legacy = store.open_path(f"{ref.state_relpath}/ganblr_model.joblib")
            if legacy.is_file():
                path = legacy
            else:
                raise FileNotFoundError(
                    f"Missing GANBLR state at {rel}; train with artifact_state_dir / artifact pipeline."
                )
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                obj = pickle.load(f)
        else:
            obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__} at {rel}, got {type(obj)}")
        return obj

    def train(self, dataset, size_category='small', *args, **kwargs):
        # parser = argparse.ArgumentParser(
        #     description="Train GANBLR and generate synthetic data")
        # parser.add_argument('--dataset', type=str, required=True,
        #                     help='Name of the dataset (e.g., adult)')
        # parser.add_argument('--size_category', type=str, required=True, choices=[
        #                     'small', 'medium', 'large'], help='Dataset size category (small/medium/large)')
        # args = parser.parse_args()

        artifact_state_dir = kwargs.pop("artifact_state_dir", None)

        epochs = kwargs.pop("train_epochs", None)
        if epochs is None:
            epochs = 150 if size_category == "large" else 100

        model_name = "ganblr"
        dataset_name = dataset
        data_dir = f"{dataset_name}"

        # Honor explicit synthetic_dir if provided (pipeline passes this)
        explicit_synth_dir = kwargs.get('synthetic_dir')
        if explicit_synth_dir and isinstance(explicit_synth_dir, str):
            save_dir = explicit_synth_dir
        else:
            save_dir = os.path.join("synthetic", dataset_name, model_name)
        os.makedirs(save_dir, exist_ok=True)

        x_train_path = os.path.join(data_dir, "x_train.csv")
        y_train_path = os.path.join(data_dir, "y_train.csv")

        # === Set Random Seed for Reproducibility ===
        seed = 42
        np.random.seed(seed)
        random.seed(seed)

        # === Load Training Data ===
        X = pd.read_csv(x_train_path)
        y = pd.read_csv(y_train_path).values.ravel()
        print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

        self.fit(X, y, k=2, epochs=epochs, batch_size=64)

        syn_data = self.sample(X.shape[0])
        df_synth = pd.DataFrame(syn_data)
        x_synth = df_synth.iloc[:, :-1]
        y_synth = df_synth.iloc[:, -1]

        x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"),
                       index=False, header=True)
        print(f"\n Synthetic data saved to: {save_dir}")

        if artifact_state_dir:
            self._save_artifact_state(artifact_state_dir)
