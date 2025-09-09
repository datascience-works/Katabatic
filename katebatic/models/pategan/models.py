# katebatic/models/pategan.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# --- If your project uses a shared base class, this import must resolve. ---
# Adjust the import path if your tree is different.
from katebatic.models.base_model import Model

"""
PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees

Reference:
  James Jordon, Jinsung Yoon, Mihaela van der Schaar,
  "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees,"
  ICLR 2019. https://openreview.net/forum?id=S1zk9iRqF7

This single file combines:
  1) A Katabatic-compatible wrapper class (PATEGANSynthesizer)
  2) A self-contained TensorFlow v1-compatible implementation of PATE-GAN

Notes:
  - We feed the GAN with the *full* [X | y] matrix so the last column is treated
    as "label" during post-processing. After generation we discretize y to the
    nearest real class and save as integers (as requested).
  - For reproducibility, we set NumPy and TF seeds (TF v1 graph).
  - Several fixes vs. the original reference code:
      * Fixed undefined loop variable when training teachers.
      * LogisticRegression configured with solver and max_iter for stability.
      * Minor typing & shape guards.
"""

# ----------------------------- Default Hyperparams -----------------------------

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_s": 1,           # student steps per outer loop
    "batch_size": 64,   # batch size
    "k": 10,            # number of teachers
    "epsilon": 1.0,     # DP epsilon
    "delta": 1e-5,      # DP delta
    "lamda": 1.0,       # Laplace noise scale (lambda)
}

# ----------------------------- Core PATE-GAN Code -----------------------------

# We keep TF v1 compat local to this module so other models aren't affected.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.linear_model import LogisticRegression


def _xavier_init(size: List[int]):
    in_dim = float(size[0])
    xavier_stddev = 1.0 / np.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def _sample_Z(m: int, n: int):
    return np.random.uniform(-1.0, 1.0, size=[m, n])


def _pate_lambda(x_vec: np.ndarray, teacher_models: List[LogisticRegression], lamda: float):
    """
    Compute the PATE vote with Laplace noise.

    Returns:
      n0, n1, out_label (0/1)
    """
    votes = []
    x_vec = np.reshape(x_vec, (1, -1))
    for teacher in teacher_models:
        pred = teacher.predict(x_vec)  # shape (1,)
        votes.append(pred[0])
    votes = np.asarray(votes)
    n0 = int(np.sum(votes == 0))
    n1 = int(np.sum(votes == 1))

    lap_noise = np.random.laplace(loc=0.0, scale=lamda)
    prob_1 = (n1 + lap_noise) / float(n0 + n1 if (n0 + n1) > 0 else 1)
    out = int(prob_1 > 0.5)
    return n0, n1, out


def _pategan_core(x_train: np.ndarray, parameters: Dict[str, Any], seed: int = 42) -> np.ndarray:
    """
    A TensorFlow v1 implementation of the basic PATE-GAN loop that *does not*
    require labels in the data (it learns to mimic x_train's distribution).
    We intentionally pass the *full* [X | y] matrix so the last column can be
    post-processed as y.

    Args:
      x_train: np.ndarray, shape [n_samples, n_features]
      parameters: dict of hyperparameters (see DEFAULT_PARAMS)
      seed: random seed for reproducibility

    Returns:
      x_train_hat: np.ndarray synthetic samples (same shape as x_train)
    """
    # Seeds
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Unpack params
    n_s = int(parameters.get("n_s", 1))
    batch_size = int(parameters.get("batch_size", 64))
    k = int(parameters.get("k", 10))
    epsilon = float(parameters.get("epsilon", 1.0))
    delta = float(parameters.get("delta", 1e-5))
    lamda = float(parameters.get("lamda", 1.0))

    # Other constants
    L = 20
    alpha = np.zeros([L], dtype=np.float64)
    epsilon_hat = 0.0

    # Shapes
    no, dim = x_train.shape
    z_dim = int(dim)
    student_h_dim = int(dim)
    generator_h_dim = int(4 * dim)

    # Partition data into k subsets
    partition_size = max(1, int(no / k))
    idx_perm = np.random.permutation(no)
    x_partition = []
    for i in range(k):
        start = i * partition_size
        end = min((i + 1) * partition_size, no)
        if start >= no:
            # In case k > no, avoid empty slices
            start = 0
            end = min(partition_size, no)
        x_partition.append(x_train[idx_perm[start:end], :])

    # Placeholders
    Y = tf.placeholder(tf.float32, shape=[None, 1])  # PATE labels (0/1)
    Z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # Student vars
    S_W1 = tf.Variable(_xavier_init([dim, student_h_dim]))
    S_b1 = tf.Variable(tf.zeros(shape=[student_h_dim]))
    S_W2 = tf.Variable(_xavier_init([student_h_dim, 1]))
    S_b2 = tf.Variable(tf.zeros(shape=[1]))
    theta_S = [S_W1, S_W2, S_b1, S_b2]

    # Generator vars
    G_W1 = tf.Variable(_xavier_init([z_dim, generator_h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[generator_h_dim]))
    G_W2 = tf.Variable(_xavier_init([generator_h_dim, generator_h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[generator_h_dim]))
    G_W3 = tf.Variable(_xavier_init([generator_h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Models
    def generator(z):
        h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
        h2 = tf.nn.tanh(tf.matmul(h1, G_W2) + G_b2)
        out = tf.nn.sigmoid(tf.matmul(h2, G_W3) + G_b3)
        return out

    def student(x):
        h1 = tf.nn.relu(tf.matmul(x, S_W1) + S_b1)
        out = tf.matmul(h1, S_W2) + S_b2  # linear score
        return out

    G_sample = generator(Z)
    S_fake = student(G_sample)

    # WGAN-style losses on student/generator (matching reference)
    S_loss = tf.reduce_mean(Y * S_fake) - tf.reduce_mean((1.0 - Y) * S_fake)
    G_loss = -tf.reduce_mean(S_fake)

    S_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-S_loss, var_list=theta_S)
    G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G)
    clip_S = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_S]

    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training loop (privacy accountant until epsilon_hat >= epsilon)
    while epsilon_hat < epsilon:
        # 1) Train k teacher models (Logistic Regression separating real vs. fake)
        teacher_models: List[LogisticRegression] = []
        for i in range(k):
            # Fresh fake batch
            Z_mb = _sample_Z(partition_size, z_dim)
            G_mb = sess.run(G_sample, feed_dict={Z: Z_mb})

            # Sample real partition
            X_part = x_partition[i % len(x_partition)]
            if len(X_part) == 0:
                # Fallback if an empty slice occurred
                X_part = x_train

            take = min(partition_size, X_part.shape[0])
            sel_idx = np.random.permutation(X_part.shape[0])[:take]
            X_mb = X_part[sel_idx, :]

            # Combine and label: real=1, fake=0
            X_comb = np.concatenate([X_mb, G_mb[:take]], axis=0)
            Y_comb = np.concatenate([np.ones((take,)), np.zeros((take,))], axis=0)

            # Stable LR config
            lr = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                n_jobs=1,
                random_state=seed,
            )
            lr.fit(X_comb, Y_comb)
            teacher_models.append(lr)

        # 2) Student training for n_s steps using PATE noisy votes
        for _ in range(n_s):
            Z_mb = _sample_Z(batch_size, z_dim)
            G_mb = sess.run(G_sample, feed_dict={Z: Z_mb})

            # Noisy votes per generated sample
            Y_mb_list = []
            for j in range(G_mb.shape[0]):
                n0, n1, r_j = _pate_lambda(G_mb[j, :], teacher_models, lamda)
                Y_mb_list.append(r_j)

                # Update privacy accountant moments (as in reference)
                q = np.log(2 + lamda * abs(n0 - n1)) - np.log(4.0) - (lamda * abs(n0 - n1))
                q = float(np.exp(q))
                # Update alphas
                for l in range(L):
                    # Two candidate bounds; take min (reference trick)
                    temp1 = 2.0 * (lamda ** 2) * (l + 1) * (l + 2)
                    temp2 = (1.0 - q) * (((1.0 - q) / (1.0 - q * np.exp(2 * lamda))) ** (l + 1)) + \
                            q * np.exp(2 * lamda * (l + 1))
                    # avoid numerical issues
                    temp2 = float(np.maximum(temp2, 1e-12))
                    alpha[l] += min(float(temp1), float(np.log(temp2)))

            Y_mb = np.asarray(Y_mb_list, dtype=np.float32).reshape(-1, 1)
            # Student update
            _, _, _ = sess.run([S_solver, S_loss, clip_S], feed_dict={Z: Z_mb, Y: Y_mb})

        # 3) Generator update (one step)
        Z_mb = _sample_Z(batch_size, z_dim)
        _ = sess.run([G_solver, G_loss], feed_dict={Z: Z_mb})

        # 4) Compute epsilon_hat from moments
        eps_list = []
        for l in range(L):
            temp_alpha = (alpha[l] + np.log(1.0 / delta)) / float(l + 1)
            eps_list.append(temp_alpha)
        epsilon_hat = float(np.min(eps_list))

    # Produce synthetic data
    X_hat = sess.run(G_sample, feed_dict={Z: _sample_Z(no, z_dim)})
    sess.close()
    return X_hat


# ------------------------- Katabatic Wrapper Class ----------------------------

class PATEGANSynthesizer(Model):
    """Katabatic-compatible PATE-GAN wrapper (single-file implementation)."""

    name = "PATE-GAN"

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {**DEFAULT_PARAMS, **params}
        self._train_full: Optional[np.ndarray] = None  # [X | y], float32
        self._synth_full: Optional[np.ndarray] = None  # [X | y], float32

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        # Add others if your env requires them at import time
        return ["tensorflow", "numpy", "pandas", "scikit-learn"]

    # ------------------------------ Train Entry ------------------------------

    def train(self, dataset: str, size_category: str = "small", *args, **kwargs):
        """
        Katabatic-style train:
          - reads {dataset}/x_train.csv and {dataset}/y_train.csv
          - runs PATE-GAN on the concatenated [X | y]
          - writes synthetic/{dataset}/pategan/{x_synth.csv, y_synth.csv}
          - ensures y_synth is DISCRETE int classes
        """
        # ---- paths ----
        model_name = "pategan"
        dataset_name = dataset
        data_dir = f"{dataset_name}"
        save_dir = os.path.join("synthetic", dataset_name, model_name)
        os.makedirs(save_dir, exist_ok=True)

        x_train_path = os.path.join(data_dir, "x_train.csv")
        y_train_path = os.path.join(data_dir, "y_train.csv")
        if not (os.path.exists(x_train_path) and os.path.exists(y_train_path)):
            raise FileNotFoundError(
                f"Expected x_train.csv & y_train.csv under {data_dir}"
            )

        # ---- seeds ----
        seed = int(kwargs.get("seed", 42))
        np.random.seed(seed)

        # ---- load ----
        X = pd.read_csv(x_train_path).to_numpy()
        y = pd.read_csv(y_train_path).values.ravel()
        if y.ndim != 1:
            y = np.ravel(y)
        print(f"[PATE-GAN] Loaded X shape: {X.shape}, y shape: {y.shape}")

        # combine [X | y] for the GAN to model full table
        train_full = np.hstack([X, y.reshape(-1, 1)]).astype(np.float32)
        self._train_full = train_full

        # ---- params (per-call overrides) ----
        run_params = dict(self.params)
        for k in ("n_s", "batch_size", "k", "epsilon", "delta", "lamda"):
            if (k in kwargs) and (kwargs[k] is not None):
                run_params[k] = kwargs[k]

        # ---- run core ----
        synth_full = _pategan_core(train_full, run_params, seed=seed).astype(np.float32)

        # ---- optional: control number of rows ----
        rows_to_generate = int(kwargs.get("rows_to_generate", X.shape[0]))
        cur = synth_full.shape[0]
        if rows_to_generate < cur:
            synth_full = synth_full[:rows_to_generate]
        elif rows_to_generate > cur:
            idx = np.random.randint(0, cur, size=rows_to_generate)
            synth_full = synth_full[idx]

        # ---- split and discretize labels ----
        X_synth = synth_full[:, :-1]
        y_float = synth_full[:, -1]

        # True classes from real y (as ints)
        real_classes = np.unique(y.astype(int))
        # Snap each synthetic y to nearest real class
        y_disc = real_classes[
            np.argmin(np.abs(y_float[:, None] - real_classes[None, :]), axis=1)
        ].astype(int)

        # ---- persist for generator API ----
        self._synth_full = np.hstack([X_synth, y_disc.reshape(-1, 1)]).astype(np.float32)
        self.is_fitted = True  # if your base class uses this flag

        # ---- SAVE (as ints, per requirement) ----
        pd.DataFrame(X_synth.astype(int)).to_csv(
            os.path.join(save_dir, "x_synth.csv"), index=False
        )
        pd.DataFrame(y_disc.astype(int)).to_csv(
            os.path.join(save_dir, "y_synth.csv"), index=False, header=True
        )

        print("[PATE-GAN] x_synth shape:", X_synth.shape, "saved as ints")
        print("[PATE-GAN] y_synth unique classes:", np.unique(y_disc))

        return self

    # ------------------------------ Generator API ------------------------------

    def generate(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X_synth, y_synth) from the last trained model."""
        if not getattr(self, "is_fitted", False) or self._synth_full is None:
            raise RuntimeError("Model is not trained yet. Call train(...) first.")

        synth = self._synth_full
        if n is not None and n > 0:
            cur = synth.shape[0]
            if n < cur:
                synth = synth[:n]
            elif n > cur:
                if seed is not None:
                    np.random.seed(seed)
                idx = np.random.randint(0, cur, size=n)
                synth = synth[idx]

        X = synth[:, :-1]
        y = synth[:, -1].astype(int)
        return X, y

    # ------------------------------ Katabatic stubs ----------------------------

    def sample(self, n: int) -> np.ndarray:
        """Return a combined [X | y] matrix for n samples (wrapper over generate)."""
        X, y = self.generate(n=n)
        return np.hstack([X, y.reshape(-1, 1)])

    def evaluate(self, real_data, metrics=None):
        """Evaluation is handled by Katabatic evaluators."""
        raise NotImplementedError("Use Katabatic's evaluation pipeline instead.")
