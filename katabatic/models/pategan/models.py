import os
from typing import Any

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model

from .utils import DataTransformer, save_metadata, set_global_seed


class PATEGAN(Model):
    """PATE-GAN aligned to the original authors' released implementation."""

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        num_teachers: int = 10,
        niter: int = 10000,
        batch_size: int = 64,
        z_dim: int | None = None,
        learning_rate: float = 1e-4,
        lamda: float = 1.0,
        n_s: int = 5,
        random_state: int = 42,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.num_teachers = num_teachers
        self.niter = niter  # Katabatic safety cap; source stops on epsilon_hat
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.n_s = n_s
        self.random_state = random_state

        self.transformer: DataTransformer | None = None
        self._sess = None
        self._G_sample = None
        self._Z = None
        self._Y = None
        self._is_built = False
        self._X_dim = None
        self.epsilon_hat = 0.0
        self.training_iterations = 0

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["tensorflow", "scipy", "sklearn", "pandas", "numpy"]

    def _xavier_init(self, size):
        import tensorflow.compat.v1 as tf

        in_dim = size[0]
        xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    @staticmethod
    def _sample_z(m: int, n: int) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=[m, n])

    @staticmethod
    def _pate_lamda(x, teacher_models, lamda):
        y_hat = []
        for teacher in teacher_models:
            temp_y = teacher.predict(np.reshape(x, [1, -1]))
            y_hat.append(temp_y)

        y_hat = np.asarray(y_hat).reshape(-1)
        n0 = int(np.sum(y_hat == 0))
        n1 = int(np.sum(y_hat == 1))

        lap_noise = np.random.laplace(loc=0.0, scale=lamda)
        out = (n1 + lap_noise) / float(n0 + n1)
        out = int(out > 0.5)
        return n0, n1, out

    def _generator(self, z):
        import tensorflow.compat.v1 as tf

        g_h1 = tf.nn.tanh(tf.matmul(z, self.G_W1) + self.G_b1)
        g_h2 = tf.nn.tanh(tf.matmul(g_h1, self.G_W2) + self.G_b2)
        return tf.nn.sigmoid(tf.matmul(g_h2, self.G_W3) + self.G_b3)

    def _student(self, x):
        import tensorflow.compat.v1 as tf

        s_h1 = tf.nn.relu(tf.matmul(x, self.S_W1) + self.S_b1)
        return tf.matmul(s_h1, self.S_W2) + self.S_b2

    def _build_model(self, x_dim: int):
        import tensorflow.compat.v1 as tf

        if self._is_built:
            return

        tf.disable_v2_behavior()
        tf.reset_default_graph()

        self._X_dim = int(x_dim)
        if self.z_dim is None:
            self.z_dim = self._X_dim

        student_h_dim = self._X_dim
        generator_h_dim = 4 * self._X_dim

        self._Y = tf.placeholder(tf.float32, shape=[None, 1])
        self._Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        self.S_W1 = tf.Variable(self._xavier_init([self._X_dim, student_h_dim]))
        self.S_b1 = tf.Variable(tf.zeros(shape=[student_h_dim]))
        self.S_W2 = tf.Variable(self._xavier_init([student_h_dim, 1]))
        self.S_b2 = tf.Variable(tf.zeros(shape=[1]))
        theta_s = [self.S_W1, self.S_W2, self.S_b1, self.S_b2]

        self.G_W1 = tf.Variable(self._xavier_init([self.z_dim, generator_h_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[generator_h_dim]))
        self.G_W2 = tf.Variable(self._xavier_init([generator_h_dim, generator_h_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[generator_h_dim]))
        self.G_W3 = tf.Variable(self._xavier_init([generator_h_dim, self._X_dim]))
        self.G_b3 = tf.Variable(tf.zeros(shape=[self._X_dim]))
        theta_g = [
            self.G_W1,
            self.G_W2,
            self.G_W3,
            self.G_b1,
            self.G_b2,
            self.G_b3,
        ]

        self._G_sample = self._generator(self._Z)
        s_fake = self._student(self._G_sample)

        self._S_loss = tf.reduce_mean(self._Y * s_fake) - tf.reduce_mean(
            (1 - self._Y) * s_fake
        )
        self._G_loss = -tf.reduce_mean(s_fake)

        self._S_solver = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate
        ).minimize(-self._S_loss, var_list=theta_s)
        self._G_solver = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate
        ).minimize(self._G_loss, var_list=theta_g)

        self._clip_S = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_s]

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._is_built = True

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        verbose: int = 1,
    ) -> "PATEGAN":
        from sklearn.linear_model import LogisticRegression

        if y is not None:
            y_df = y.to_frame() if isinstance(y, pd.Series) else y.copy()
            data = pd.concat(
                [X.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1
            )
        else:
            data = X.reset_index(drop=True).copy()

        if self.num_teachers < 1:
            raise ValueError("num_teachers must be at least 1")
        if self.lamda <= 0:
            raise ValueError("lamda must be greater than 0")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be between 0 and 1")

        set_global_seed(self.random_state)
        np.random.seed(self.random_state)

        self.transformer = DataTransformer()
        x_encoded = self.transformer.fit_transform(data)

        no, dim = x_encoded.shape
        if self.num_teachers > no:
            raise ValueError("num_teachers cannot exceed number of training rows")

        self._build_model(dim)

        partition_data_no = int(no / self.num_teachers)
        if partition_data_no < 1:
            raise ValueError("Each teacher requires at least one training row")

        idx = np.random.permutation(no)
        x_partition = []
        for i in range(self.num_teachers):
            temp_idx = idx[
                int(i * partition_data_no) : int((i + 1) * partition_data_no)
            ]
            x_partition.append(x_encoded[temp_idx, :])

        L = 20
        alpha = np.zeros([L])
        epsilon_hat = 0.0
        self.training_iterations = 0

        if verbose:
            print(f"Training PATE-GAN with ε={self.epsilon}, δ={self.delta}")
            print(
                f"Teachers: {self.num_teachers}, batch_size: {self.batch_size}, "
                f"n_s: {self.n_s}, lambda: {self.lamda}"
            )

        while epsilon_hat < self.epsilon and self.training_iterations < self.niter:
            teacher_models = []

            # Released source has a stale-index bug here. Using i is the
            # paper-consistent correction: teacher i trains on partition i.
            for i in range(self.num_teachers):
                z_mb = self._sample_z(partition_data_no, self.z_dim)
                g_mb = self._sess.run(self._G_sample, feed_dict={self._Z: z_mb})

                temp_x = x_partition[i]
                perm = np.random.permutation(len(temp_x))
                x_mb = temp_x[perm[:partition_data_no], :]

                x_comb = np.concatenate((x_mb, g_mb), axis=0)
                y_comb = np.concatenate(
                    (
                        np.ones([partition_data_no]),
                        np.zeros([partition_data_no]),
                    ),
                    axis=0,
                )

                teacher = LogisticRegression()
                teacher.fit(x_comb, y_comb)
                teacher_models.append(teacher)

            for _ in range(self.n_s):
                z_mb = self._sample_z(self.batch_size, self.z_dim)
                g_mb = self._sess.run(self._G_sample, feed_dict={self._Z: z_mb})

                y_mb = []
                for j in range(self.batch_size):
                    n0, n1, r_j = self._pate_lamda(
                        g_mb[j, :], teacher_models, self.lamda
                    )
                    y_mb.append(r_j)

                    q = (
                        np.log(2 + self.lamda * abs(n0 - n1))
                        - np.log(4.0)
                        - self.lamda * abs(n0 - n1)
                    )
                    q = np.exp(q)

                    for order in range(L):
                        temp1 = 2 * (self.lamda**2) * (order + 1) * (order + 2)
                        denominator = 1 - q * np.exp(2 * self.lamda)

                        if denominator <= 0:
                            temp2_log = np.inf
                        else:
                            temp2 = (1 - q) * (
                                ((1 - q) / denominator) ** (order + 1)
                            ) + q * np.exp(2 * self.lamda * (order + 1))
                            temp2_log = np.log(temp2) if temp2 > 0 else np.inf

                        alpha[order] += np.min([temp1, temp2_log])

                y_mb = np.reshape(np.asarray(y_mb), [-1, 1])
                self._sess.run(
                    [self._S_solver, self._S_loss, self._clip_S],
                    feed_dict={self._Z: z_mb, self._Y: y_mb},
                )

            z_mb = self._sample_z(self.batch_size, self.z_dim)
            self._sess.run(
                [self._G_solver, self._G_loss],
                feed_dict={self._Z: z_mb},
            )

            curr_list = []
            for order in range(L):
                temp_alpha = (alpha[order] + np.log(1 / self.delta)) / float(order + 1)
                curr_list.append(temp_alpha)

            epsilon_hat = float(np.min(curr_list))
            self.training_iterations += 1

            if verbose and self.training_iterations % 10 == 0:
                print(
                    f"Iteration {self.training_iterations}: "
                    f"epsilon_hat={epsilon_hat:.6f}"
                )

        self.epsilon_hat = epsilon_hat
        self.is_fitted = True

        if verbose:
            if epsilon_hat >= self.epsilon:
                print(
                    f"Privacy stopping condition reached: epsilon_hat={epsilon_hat:.6f}"
                )
            else:
                print(
                    "niter safety cap reached before privacy stopping condition; "
                    f"epsilon_hat={epsilon_hat:.6f}"
                )
            print("Training completed!")

        return self

    def sample(
        self,
        n: int,
        conditional: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling")
        if n < 1:
            raise ValueError("n must be at least 1")
        if conditional is not None:
            print(
                "Warning: conditional sampling is not part of the original "
                "PATE-GAN implementation"
            )

        all_samples = []
        n_batches = int(np.ceil(n / self.batch_size))
        for i in range(n_batches):
            batch_size = min(self.batch_size, n - i * self.batch_size)
            z_sample = self._sample_z(batch_size, self.z_dim)
            x_synth = self._sess.run(self._G_sample, feed_dict={self._Z: z_sample})
            all_samples.append(x_synth)

        x_synth_all = np.vstack(all_samples)[:n]
        return self.transformer.inverse_transform(x_synth_all)

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: str | None = None,
        **kwargs,
    ) -> "PATEGAN":
        if "epsilon" in kwargs:
            self.epsilon = kwargs.pop("epsilon")
        if "delta" in kwargs:
            self.delta = kwargs.pop("delta")
        if "num_teachers" in kwargs:
            self.num_teachers = kwargs.pop("num_teachers")
        if "niter" in kwargs:
            self.niter = kwargs.pop("niter")
        if "batch_size" in kwargs:
            self.batch_size = kwargs.pop("batch_size")
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs.pop("learning_rate")
        if "lamda" in kwargs:
            self.lamda = kwargs.pop("lamda")
        if "lambda_noise" in kwargs:
            self.lamda = kwargs.pop("lambda_noise")
        if "n_s" in kwargs:
            self.n_s = kwargs.pop("n_s")
        if "z_dim" in kwargs:
            self.z_dim = kwargs.pop("z_dim")
        if "random_state" in kwargs:
            self.random_state = kwargs.pop("random_state")

        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")

        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"x_train.csv not found in {dataset_dir}")

        x_train = pd.read_csv(x_train_path)
        y_train = None
        if os.path.exists(y_train_path):
            y_train = pd.read_csv(y_train_path)
            if isinstance(y_train, pd.DataFrame) and len(y_train.columns) == 1:
                y_train = y_train.iloc[:, 0]

        print(f"Loaded training data: X shape={x_train.shape}", end="")
        if y_train is not None:
            print(f", y shape={(len(y_train),)}")
        else:
            print()

        self.fit(x_train, y_train, verbose=kwargs.get("verbose", 1))

        n_samples = len(x_train)
        print(f"\nGenerating {n_samples} synthetic samples...")
        df_synth = self.sample(n_samples)

        if y_train is not None:
            target_cols = (
                y_train.columns.tolist()
                if isinstance(y_train, pd.DataFrame)
                else [y_train.name]
            )
            x_synth = df_synth.drop(columns=target_cols)
            y_synth = df_synth[target_cols].copy()
        else:
            x_synth = df_synth
            y_synth = None

        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)

            x_synth_path = os.path.join(synthetic_dir, "x_synth.csv")
            x_synth.to_csv(x_synth_path, index=False)
            print(f"Saved x_synth.csv to {x_synth_path}")

            if y_synth is not None:
                y_synth_path = os.path.join(synthetic_dir, "y_synth.csv")
                y_synth.to_csv(y_synth_path, index=False)
                print(f"Saved y_synth.csv to {y_synth_path}")

            metadata_path = os.path.join(synthetic_dir, "metadata.json")
            training_config = {
                "source_alignment": "original_authors_released_code",
                "niter_safety_cap": self.niter,
                "training_iterations": self.training_iterations,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "n_s": self.n_s,
                "z_dim": self.z_dim,
                "optimizer": "RMSProp",
                "student_weight_clip": 0.01,
            }
            privacy_config = {
                "epsilon_target": self.epsilon,
                "epsilon_hat": self.epsilon_hat,
                "delta": self.delta,
                "num_teachers": self.num_teachers,
                "lamda": self.lamda,
                "moments_accountant_L": 20,
            }
            save_metadata(
                metadata_path,
                self.transformer,
                training_config,
                privacy_config,
                self.random_state,
            )
            print(f"Saved metadata.json to {metadata_path}")

        return self

    def evaluate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        model: str = "lr",
        metrics: list | None = None,
        task: str | None = None,
        random_state: int = 42,
        **kwargs,
    ) -> dict[str, float]:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )

        if task is None:
            task = (
                "classification"
                if y.dtype in ["object", "category"] or y.nunique() < 20
                else "regression"
            )

        df_synth = self.sample(len(x))
        x_synth = df_synth.iloc[:, :-1]
        y_synth = df_synth.iloc[:, -1]

        if task == "classification":
            if model == "rf":
                estimator = RandomForestClassifier(
                    random_state=random_state, n_estimators=100
                )
            else:
                estimator = LogisticRegression(random_state=random_state, max_iter=1000)

            estimator.fit(x_synth, y_synth)
            y_pred = estimator.predict(x)
            results = {
                "accuracy": accuracy_score(y, y_pred),
                "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            }
            if y.nunique() == 2:
                try:
                    y_proba = estimator.predict_proba(x)[:, 1]
                    results["roc_auc"] = roc_auc_score(y, y_proba)
                except (AttributeError, ValueError):
                    pass
        else:
            if model == "rf":
                estimator = RandomForestRegressor(
                    random_state=random_state, n_estimators=100
                )
            else:
                estimator = LinearRegression()

            estimator.fit(x_synth, y_synth)
            y_pred = estimator.predict(x)
            results = {
                "r2": r2_score(y, y_pred),
                "mae": mean_absolute_error(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            }

        return results

    def __del__(self):
        if self._sess is not None:
            try:
                self._sess.close()
            except Exception:
                pass
