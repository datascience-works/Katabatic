import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from katabatic.models.base_model import Model

from .utils import elr_loss, get_lr, sample, softmax_weight


class GANBLR(Model):
    """
    The GANBLR Model.
    """

    ARTIFACT_STATE_FILES = ("ganblr_model.pkl",)

    def __init__(self) -> None:
        super().__init__()
        self.check_dependencies()  # Check dependencies on initialization
        self._d = None
        self.__gen_weights = None
        self.batch_size = None
        self.epochs = 150
        self.k = None
        self.constraints = None
        self._ordinal_encoder = OrdinalEncoder(
            dtype=int, handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._label_encoder = LabelEncoder()
        self._feature_names = None
        self._target_name = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return a list of required dependencies for this model."""
        return ["tensorflow", "pgmpy", "sklearn", "scipy"]

    def fit(self, x, y, k=0, batch_size=32, epochs=150, warmup_epochs=1, verbose=1):
        """
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
        """
        epsilon = 1e-10

        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        self._feature_names = (
            list(x.columns)
            if hasattr(x, "columns")
            else [f"feature_{i}" for i in range(np.asarray(x).shape[1])]
        )
        self._target_name = y.name if hasattr(y, "name") and y.name else "target"
        x = self._ordinal_encoder.fit_transform(x)
        y = self._label_encoder.fit_transform(y).astype(int)
        d = DataUtils(x, y)
        self._d = d
        self.k = k
        self.batch_size = batch_size
        if verbose:
            print("warmup run:")

        self._warmup_run(warmup_epochs, verbose=verbose)
        sample_size = min(2000, d.data_size)
        # print("Datasize and min" , d.data_size)
        syn_data = self._sample(size=sample_size, verbose=0)
        discriminator_label = np.hstack([np.ones(d.data_size), np.zeros(sample_size)])
        # [np.ones(d.data_size), np.zeros(d.data_size)])
        # build discriminator once before each epoch run.
        disc = self._discrim()
        for i in range(epochs):
            discriminator_input = np.vstack([x, syn_data[:, :-1]])
            disc_input, disc_label = sample(
                discriminator_input, discriminator_label, frac=0.8
            )
            # disc = self._discrim()
            d_history = disc.fit(
                disc_input, disc_label, batch_size=batch_size, epochs=1, verbose=0
            ).history
            prob_fake = disc.predict(x, verbose=0)
            # ls = np.mean(-np.log(np.subtract(1, prob_fake)))
            ls = np.mean(-np.log(np.clip(1 - prob_fake, epsilon, 1)))
            g_history = self._run_generator(loss=ls).history
            syn_data = self._sample(size=sample_size, verbose=0)

            if verbose:
                print(
                    f"Epoch {i + 1}/{epochs}: G_loss = {g_history['loss'][0]:.6f}, G_accuracy = {g_history['accuracy'][0]:.6f}, D_loss = {d_history['loss'][0]:.6f}, D_accuracy = {d_history['accuracy'][0]:.6f}"
                )

        self.is_fitted = True
        return self

    def evaluate(self, x, y, model="lr") -> float:
        """
        Perform a TSTR(Training on Synthetic data, Testing on Real data) evaluation.

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

        eval_model = None
        models = dict(
            lr=LogisticRegression, rf=RandomForestClassifier, mlp=MLPClassifier
        )
        if model in models.keys():
            eval_model = models[model]()
        elif hasattr(model, "fit") and hasattr(model, "predict"):
            eval_model = model
        else:
            raise Exception(
                "Invalid Arugument `model`, Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method."
            )

        synthetic_data = self._sample()
        synthetic_x, synthetic_y = synthetic_data[:, :-1], synthetic_data[:, -1]
        x_test = self._ordinal_encoder.transform(x)
        y_test = self._label_encoder.transform(y)

        categories = self._d.get_categories()
        pipline = Pipeline(
            [
                (
                    "encoder",
                    OneHotEncoder(categories=categories, handle_unknown="ignore"),
                ),
                ("model", eval_model),
            ]
        )
        pipline.fit(synthetic_x, synthetic_y)
        pred = pipline.predict(x_test)
        return accuracy_score(y_test, pred)

    def sample(self, size=None, verbose=1, seed=None) -> pd.DataFrame:
        """
        Generate synthetic data.

        Parameters
        ----------
        size : int or None
            Size of the data to be generated. set to `None` to make the size equal to the size of the training set.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        seed : int or None, default=None
            Random seed for reproducible sampling. Pass an int to get deterministic
            output across calls; leave as None for unseeded (non-reproducible) sampling.

        Return:
        -----------------
        synthetic_samples : pd.DataFrame
            Generated synthetic data, with the same column names used at fit time.
        """
        ordinal_data = self._sample(size, verbose, seed=seed)
        origin_x = self._ordinal_encoder.inverse_transform(ordinal_data[:, :-1])
        origin_y = self._label_encoder.inverse_transform(ordinal_data[:, -1]).reshape(
            -1, 1
        )
        columns = (
            self._feature_names or [f"feature_{i}" for i in range(origin_x.shape[1])]
        ) + [self._target_name or "target"]
        return pd.DataFrame(np.hstack([origin_x, origin_y]), columns=columns)

    def _sample(self, size=None, verbose=1, seed=None) -> np.ndarray:
        """
        Generate synthetic data in ordinal encoding format
        """
        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        if seed is not None:
            np.random.seed(seed)
        # basic varibles
        d = self._d
        feature_cards = np.array(d.feature_uniques)
        # ensure sum of each constraint group equals to 1, then re concat the probs
        _idxs = np.cumsum([0] + d._kdbe.constraints_.tolist())
        constraint_idxs = [(_idxs[i], _idxs[i + 1]) for i in range(len(_idxs) - 1)]

        probs = np.exp(self.__gen_weights[0])
        cpd_probs = [probs[start:end, :] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p / p.sum(axis=0) for p in cpd_probs])

        # assign the probs to the full cpd tables
        idxs = np.cumsum([0] + d._kdbe.high_order_feature_uniques_)
        feature_idxs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        have_value_idxs = d._kdbe.have_value_idxs_
        full_cpd_probs = []
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            # (n_high_order_feature_uniques, n_classes)
            cpd_prob_ = cpd_probs[start:end, :]
            # (n_all_combination) Note: the order is (*parent, variable)
            have_value_ravel = have_value.ravel()
            # (n_classes * n_all_combination)
            have_value_ravel_repeat = np.hstack([have_value_ravel] * d.num_classes)
            # (n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
            full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
            full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
            # (n_classes * n_parent_combinations, n_variable_unique)
            full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
            full_cpd_prob = _add_uniform(full_cpd_prob, noise=0)
            full_cpd_probs.append(full_cpd_prob)

        # prepare node and edge names
        node_names = [str(i) for i in range(d.num_features + 1)]
        edge_names = [(str(i), str(j)) for i, j in d._kdbe.edges_]
        y_name = node_names[-1]

        # create TabularCPD objects
        evidences = d._kdbe.dependencies_
        feature_cpds = [
            TabularCPD(
                str(name),
                feature_cards[name],
                table,
                evidence=[y_name, *[str(e) for e in evidences]],
                evidence_card=[d.num_classes, *feature_cards[evidences].tolist()],
            )
            for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
        ]
        y_probs = (d.class_counts / d.data_size).reshape(-1, 1)
        y_cpd = TabularCPD(y_name, d.num_classes, y_probs)

        # create kDB model, then sample the data
        model = DiscreteBayesianNetwork(edge_names)
        model.add_cpds(y_cpd, *feature_cpds)
        sample_size = d.data_size if size is None else size
        result = BayesianModelSampling(model).forward_sample(
            size=sample_size, show_progress=verbose > 0
        )
        sorted_result = result[node_names].values

        return sorted_result

    def _warmup_run(self, epochs, verbose=None):
        d = self._d
        tf.keras.backend.clear_session()
        ohex = d.get_kdbe_x(self.k)
        self.constraints = softmax_weight(d.constraint_positions)
        elr = get_lr(ohex.shape[1], d.num_classes, self.constraints)
        history = elr.fit(
            ohex, d.y, batch_size=self.batch_size, epochs=epochs, verbose=verbose
        )
        self.__gen_weights = elr.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _run_generator(self, loss):
        d = self._d
        ohex = d.get_kdbe_x(self.k)
        tf.keras.backend.clear_session()
        # Adding shape to sequential, as best practise and will remove warnings
        # https://keras.io/guides/sequential_model/
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(ohex.shape[1],)))
        model.add(
            tf.keras.layers.Dense(
                d.num_classes, activation="softmax", kernel_constraint=self.constraints
            )
        )
        model.compile(loss=elr_loss(loss), optimizer="adam", metrics=["accuracy"])
        model.set_weights(self.__gen_weights)
        history = model.fit(ohex, d.y, batch_size=self.batch_size, epochs=1, verbose=0)
        self.__gen_weights = model.get_weights()
        # keep session cache to speed up GAN

        return history

    def _discrim(self):
        model = tf.keras.Sequential()
        # keras prefers advanced info on the shape of data
        model.add(tf.keras.Input(shape=(self._d.num_features,)))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def train(self, dataset, size_category="small", *args, **kwargs):
        # parser = argparse.ArgumentParser(
        #     description="Train GANBLR and generate synthetic data")
        # parser.add_argument('--dataset', type=str, required=True,
        #                     help='Name of the dataset (e.g., adult)')
        # parser.add_argument('--size_category', type=str, required=True, choices=[
        #                     'small', 'medium', 'large'], help='Dataset size category (small/medium/large)')
        # args = parser.parse_args()

        epochs = kwargs.get(
            "train_epochs",
            kwargs.get("epochs", 150 if size_category == "large" else 100),
        )

        model_name = "ganblr"
        dataset_name = dataset
        data_dir = f"{dataset_name}"

        # Honor explicit synthetic_dir if provided (pipeline passes this)
        explicit_synth_dir = kwargs.get("synthetic_dir")
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
        y_df = pd.read_csv(y_train_path)
        y = y_df.iloc[:, 0]
        print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

        self.fit(
            X, y, k=2, epochs=epochs, batch_size=512
        )  # increase the batch size from 64

        df_synth = self.sample(X.shape[0])
        x_synth = df_synth.iloc[:, :-1]
        y_synth = df_synth.iloc[:, -1]

        x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"), index=False, header=True)
        print(f"\n Synthetic data saved to: {save_dir}")

        # persist model state for artifact reload
        artifact_state_dir = kwargs.get("artifact_state_dir")
        if artifact_state_dir:
            self._save_artifact_state(artifact_state_dir)

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        """
        Persist fitted state so the model can be rebuilt by load_from_ref().

        Called from train() when the pipeline injects artifact_state_dir, which
        it does for any model class declaring ARTIFACT_STATE_FILES.
        """
        os.makedirs(artifact_state_dir, exist_ok=True)
        target = os.path.join(artifact_state_dir, self.ARTIFACT_STATE_FILES[0])
        with open(target, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load_from_ref(cls, store, ref) -> "GANBLR":
        """
        Rehydrate a fitted GANBLR from a versioned artifact.

        train() pickles the whole fitted instance, so reloading is a plain
        unpickle rather than the attribute/weight split CTGAN needs.
        """
        state_file = cls.ARTIFACT_STATE_FILES[0]
        state_path = store.open_path(f"{ref.state_relpath}/{state_file}")

        if not state_path.is_file():
            raise FileNotFoundError(
                f"No GANBLR state at {state_path}. The model must be trained "
                f"through a pipeline that passes artifact_state_dir."
            )

        with open(state_path, "rb") as fh:
            # loading our own saved model artifact
            instance = pickle.load(fh)  # nosec B301

        if not isinstance(instance, cls):
            raise TypeError(
                f"Artifact at {state_path} holds {type(instance).__name__}, not {cls.__name__}."
            )
        return instance


class DataUtils:
    """
    useful data utils for the preparation before training.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.data_size = len(x)
        self.num_features = x.shape[1]

        yunique, ycounts = np.unique(y, return_counts=True)
        self.num_classes = len(yunique)
        self.class_counts = ycounts
        self.feature_uniques = [
            len(np.unique(x[:, i])) for i in range(self.num_features)
        ]

        self.constraint_positions = None
        self._kdbe = None

        self.__kdbe_x = None

    def get_categories(self, idxs=None):
        if idxs is not None:
            return [self._kdbe.ohe_.categories_[i] for i in idxs]
        return self._kdbe.ohe_.categories_

    def get_kdbe_x(self, k=0, dense_format=False) -> np.ndarray:
        if self.__kdbe_x is not None:
            return self.__kdbe_x
        if self._kdbe is None:
            self._kdbe = KdbHighOrderFeatureEncoder()
            self._kdbe.fit(self.x, self.y, k=k)
        kdbex = self._kdbe.transform(self.x)
        if dense_format:
            kdbex = kdbex.todense()
        self.__kdbe_x = kdbex
        self.constraint_positions = self._kdbe.constraints_
        return kdbex

    def clear(self):
        self._kdbe = None
        self.__kdbe_x = None


def build_graph(X, y, k=2):
    """
    kDB algorithm

    Param:
    ----------------------

    Return:
    ----------------------
    graph edges
    """
    # ensure data
    num_features = X.shape[1]
    x_nodes = list(range(num_features))
    y_node = num_features

    # util func
    def _x(i):
        return X[:, i]

    def _x2comb(i, j):
        return X[:, i], X[:, j]

    # feature indexes desc sort by mutual information
    sorted_feature_idxs = np.argsort(
        [drv.information_mutual(_x(i), y) for i in range(num_features)]
    )[::-1]

    # start building graph
    edges = []
    for iter, target_idx in enumerate(sorted_feature_idxs):
        target_node = x_nodes[target_idx]
        edges.append((y_node, target_node))

        parent_candidate_idxs = sorted_feature_idxs[:iter]
        if iter <= k:
            for idx in parent_candidate_idxs:
                edges.append((x_nodes[idx], target_node))
        else:
            first_k_parent_mi_idxs = np.argsort(
                [
                    drv.information_mutual_conditional(*_x2comb(i, target_idx), y)
                    for i in parent_candidate_idxs
                ]
            )[::-1][:k]
            first_k_parent_idxs = parent_candidate_idxs[first_k_parent_mi_idxs]

            for parent_idx in first_k_parent_idxs:
                edges.append((x_nodes[parent_idx], target_node))
    return edges


# def draw_graph(edges):
#   '''
#   Draw the graph
#
#   Param
#   -----------------
#   edges: edges of the graph
#
#   '''
#   graph = nx.DiGraph(edges)
#   pos=nx.spiral_layout(graph)
#   nx.draw(graph, pos, node_color='r', edge_color='b')
#   nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")


def get_cross_table(*cols, apply_wt=False):
    """
    author: alexland

    returns:
      (i) xt, NumPy array storing the xtab results, number of dimensions is equal to
          the len(args) passed in
      (ii) unique_vals_all_cols, a tuple of 1D NumPy array for each dimension
          in xt (for a 2D xtab, the tuple comprises the row and column headers)
      pass in:
        (i) 1 or more 1D NumPy arrays of integers
        (ii) if wts is True, then the last array in cols is an array of weights

    if return_inverse=True, then np.unique also returns an integer index
    (from 0, & of same len as array passed in) such that, uniq_vals[idx] gives the original array passed in
    higher dimensional cross tabulations are supported (eg, 2D & 3D)
    cross tabulation on two variables (columns):
    >>> q1 = np.array([7, 8, 8, 8, 5, 6, 4, 6, 6, 8, 4, 6, 6, 6, 6, 8, 8, 5, 8, 6])
    >>> q2 = np.array([6, 4, 6, 4, 8, 8, 4, 8, 7, 4, 4, 8, 8, 7, 5, 4, 8, 4, 4, 4])
    >>> uv, xt = xtab(q1, q2)
    >>> uv
      (array([4, 5, 6, 7, 8]), array([4, 5, 6, 7, 8]))
    >>> xt
      array([[2, 0, 0, 0, 0],
             [1, 0, 0, 0, 1],
             [1, 1, 0, 2, 4],
             [0, 0, 1, 0, 0],
             [5, 0, 1, 0, 1]], dtype=uint64)
    """
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")

    if len(cols) == 0:
        raise TypeError("xtab() requires at least one argument")

    def fnx1(c):
        return len(c.squeeze().shape)

    if not all([fnx1(col) == 1 for col in cols]):
        raise ValueError("all input arrays must be 1D")

    if apply_wt:
        cols, wt = cols[:-1], cols[-1]
    else:
        wt = 1

    uniq_vals_all_cols, idx = zip(
        *(np.unique(col, return_inverse=True) for col in cols)
    )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    dtype_xt = "float" if apply_wt else "uint"
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt


def _get_dependencies_without_y(variables, y_name, kdb_edges):
    """
    evidences of each variable without y.

    Param:
    --------------
    variables: variable names

    y_name: class name

    kdb_edges: list of tuple (source, target)
    """
    dependencies = {}
    kdb_edges_without_y = [edge for edge in kdb_edges if edge[0] != y_name]
    mi_desc_order = {t: i for i, (s, t) in enumerate(kdb_edges) if s == y_name}
    for x in variables:
        current_dependencies = [s for s, t in kdb_edges_without_y if t == x]
        if len(current_dependencies) >= 2:
            sort_dict = {t: mi_desc_order[t] for t in current_dependencies}
            dependencies[x] = sorted(sort_dict)
        else:
            dependencies[x] = current_dependencies
    return dependencies


def _add_uniform(array, noise=1e-5):
    """
    if no count on particular condition for any feature, give a uniform prob rather than leave 0
    """
    sum_by_col = np.sum(array, axis=0)
    zero_idxs = (array == 0).astype(int)
    # zero_count_by_col = np.sum(zero_idxs,axis=0)
    nunique = array.shape[0]
    result = np.zeros_like(array, dtype="float")
    for i in range(array.shape[1]):
        if sum_by_col[i] == 0:
            result[:, i] = array[:, i] + 1.0 / nunique
        elif noise != 0:
            result[:, i] = array[:, i] + noise * zero_idxs[:, i]
        else:
            result[:, i] = array[:, i]
    return result


def _normalize_by_column(array):
    sum_by_col = np.sum(array, axis=0)
    return np.divide(
        array,
        sum_by_col,
        out=np.zeros_like(array, dtype="float"),
        where=sum_by_col != 0,
    )


def _smoothing(cct, d):
    """
    probability smoothing for kdb

    Parameters:
    -----------
    cct (np.ndarray): cross count table with shape (x0, *parents)

    d (int): dimension of cct

    Return:
    --------
    smoothed joint prob table
    """
    # covert cross-count-table to joint-prob-table by doing a normalization alone axis 0
    jpt = _normalize_by_column(cct)
    smoothing_idx = jpt == 0
    if d > 1 and np.sum(smoothing_idx) > 0:
        parent = cct.sum(axis=-1)
        parent = _smoothing(parent, d - 1)
        parent_extend = parent.repeat(jpt.shape[-1]).reshape(jpt.shape)
        jpt[smoothing_idx] = parent_extend[smoothing_idx]
    return jpt


def get_high_order_feature(X, col, evidence_cols, feature_uniques):
    """
    encode the high order feature of X[col] given evidences X[evidence_cols].
    """
    if evidence_cols is None or len(evidence_cols) == 0:
        return X[:, [col]]
    else:
        evidence = [X[:, _col] for _col in evidence_cols]  # noqa: F841

        # [1, variable_unique, evidence_unique]
        base = [1, feature_uniques[col]] + [
            feature_uniques[_col] for _col in evidence_cols[::-1][:-1]
        ]
        cum_base = np.cumprod(base)[::-1]

        cols = evidence_cols + [col]
        high_order_feature = np.sum(X[:, cols] * cum_base, axis=1).reshape(-1, 1)
        return high_order_feature


def get_high_order_constraints(X, col, evidence_cols, feature_uniques):
    """
    find the constraints infomation for the high order feature X[col] given evidences X[evidence_cols].

    Returns:
    ---------------------
    tuple(have_value, high_order_uniques)

    have_value: a k+1 dimensions numpy ndarray of type boolean.
        Each dimension correspond to a variable, with the order (*evidence_cols, col)
        True indicate the corresponding combination of variable values cound be found in the dataset.
        False indicate not.

    high_order_constraints: a 1d nummy ndarray of type int.
        Each number `c` indicate that there are `c` cols shound be applying the constraints since the last constrant position(or index 0),
        in sequence.

    """
    if evidence_cols is None or len(evidence_cols) == 0:
        unique = feature_uniques[col]
        return np.ones(unique, dtype=bool), np.array([unique])
    else:
        cols = evidence_cols + [col]
        cross_table_idxs, cross_table = get_cross_table(*[X[:, i] for i in cols])
        have_value = cross_table != 0

        have_value_reshape = have_value.reshape(-1, have_value.shape[-1])
        # have_value_split = np.split(have_value_reshape, have_value_reshape.shape[0], 0)
        high_order_constraints = np.sum(have_value_reshape, axis=-1)

        return have_value, high_order_constraints


class KdbHighOrderFeatureEncoder:
    """
    High order feature encoder that uses the kdb model to retrieve the dependencies between features.

    """

    def __init__(self):
        self.dependencies_ = {}
        self.constraints_ = np.array([])
        self.have_value_idxs_ = []
        self.feature_uniques_ = []
        self.high_order_feature_uniques_ = []
        self.edges_ = []
        self.ohe_ = None
        self.k = None
        # self.full_=True

    def fit(self, X, y, k=0):
        """
        Fit the KdbHighOrderFeatureEncoder to X, y.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            data to fit in the encoder.

        y : array_like of shape (n_samples,)
            label to fit in the encoder.

        k : int, default=0
            k value of the order of the high-order feature. k = 0 will lead to a OneHotEncoder.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        self.k = k
        edges = build_graph(X, y, k)
        # n_classes = len(np.unique(y))
        num_features = X.shape[1]

        if k > 0:
            dependencies = _get_dependencies_without_y(
                list(range(num_features)), num_features, edges
            )
        else:
            dependencies = {x: [] for x in range(num_features)}

        self.dependencies_ = dependencies
        self.feature_uniques_ = [len(np.unique(X[:, i])) for i in range(num_features)]
        self.edges_ = edges
        # self.full_ = full

        Xk, constraints, have_value_idxs = self.transform(
            X, return_constraints=True, use_ohe=False
        )

        self.ohe_ = OneHotEncoder().fit(Xk)
        self.high_order_feature_uniques_ = [len(c) for c in self.ohe_.categories_]
        self.constraints_ = constraints
        self.have_value_idxs_ = have_value_idxs
        return self

    def transform(self, X, return_constraints=False, use_ohe=True):
        """
        Transform X to the high-order features.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data to fit in the encoder.

        return_constraints : bool, default=False
            Whether to return the constraint informations.

        use_ohe : bool, default=True
            Whether to transform output to one-hot format.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed input.
        """
        Xk = []
        have_value_idxs = []
        constraints = []
        for k, v in self.dependencies_.items():
            xk = get_high_order_feature(X, k, v, self.feature_uniques_)
            Xk.append(xk)

            if return_constraints:
                idx, constraint = get_high_order_constraints(
                    X, k, v, self.feature_uniques_
                )
                have_value_idxs.append(idx)
                constraints.append(constraint)

        Xk = np.hstack(Xk)
        from sklearn.preprocessing import OrdinalEncoder

        Xk = OrdinalEncoder().fit_transform(Xk)
        if use_ohe:
            Xk = self.ohe_.transform(Xk)

        if return_constraints:
            concated_constraints = np.hstack(constraints)
            return Xk, concated_constraints, have_value_idxs
        else:
            return Xk

    def fit_transform(self, X, y, k=0, return_constraints=False):
        """
        Fit KdbHighOrderFeatureEncoder to X, y, then transform X.

        Equivalent to fit(X, y, k).transform(X, return_constraints) but more convenient.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            data to fit in the encoder.

        y : array_like of shape (n_samples,)
            label to fit in the encoder.

        k : int, default=0
            k value of the kdb model. k = 0 will lead to a OneHotEncoder.

        return_constraints : bool, default=False
            whether to return the constraint informations.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed input.
        """
        return self.fit(X, y, k).transform(X, return_constraints)
