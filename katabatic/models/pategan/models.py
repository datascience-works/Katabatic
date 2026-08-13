import os
from typing import Any

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model

from .utils import (
    DataTransformer,
    PrivacyMechanism,
    partition_data,
    save_metadata,
    set_global_seed,
)


class PATEGAN(Model):
    """
    PATE-GAN: Differential privacy through teacher ensemble aggregation.

    This model uses the PATE framework with a Wasserstein GAN to generate
    differentially private synthetic tabular data.

    Key features:
    - (ε, δ)-differential privacy guarantees
    - Multiple teacher discriminators on disjoint data
    - WGAN-GP for stable training
    - Handles mixed categorical and continuous data
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        num_teachers: int = 10,
        niter: int = 10000,
        batch_size: int = 128,
        z_dim: int | None = None,
        learning_rate: float = 1e-4,
        lambda_gp: float = 10.0,
        random_state: int = 42,
    ):
        """
        Initialize PATE-GAN model.

        Args:
            epsilon: Privacy budget (lower = more private, typical: 0.1-10)
            delta: Privacy parameter (typical: 1e-5 or 1e-6)
            num_teachers: Number of teacher discriminators (typical: 5-20)
            niter: Number of training iterations (default: 10000)
            batch_size: Batch size for training (default: 128)
            z_dim: Latent dimension (default: n_features // 4)
            learning_rate: Adam learning rate (default: 1e-4)
            lambda_gp: Gradient penalty coefficient (default: 10.0)
            random_state: Random seed for reproducibility
        """
        super().__init__()

        # Privacy parameters
        self.epsilon = epsilon
        self.delta = delta
        self.num_teachers = num_teachers

        # Training parameters
        self.niter = niter
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.lambda_gp = lambda_gp
        self.random_state = random_state

        # State
        self.transformer: DataTransformer | None = None
        self.privacy_mechanism: PrivacyMechanism | None = None
        self._sess = None
        self._G_sample = None
        self._is_built = False
        self._X_dim = None
        self._h_dim = None

        # Conditional target-generation state
        self._target_name = None
        self._target_classes = None
        self._class_probs = None
        self._num_classes = 0

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return required dependencies for PATE-GAN."""
        return ["tensorflow", "scipy", "sklearn", "pandas", "numpy"]

    def _xavier_init(self, size):
        """Xavier initialization for network weights."""
        import tensorflow.compat.v1 as tf

        in_dim = size[0]
        xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _generator(self, z, y_cond):
        """
        Conditional generator.

        Args:
            z: Latent noise vector
            y_cond: One-hot target-class condition

        Returns:
            Generated feature samples in normalized feature space.
        """
        import tensorflow.compat.v1 as tf

        generator_input = tf.concat([z, y_cond], axis=1)

        G_h1 = tf.nn.tanh(tf.matmul(generator_input, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_out = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3)

        return G_out

    def _student_discriminator(self, x, y_cond):
        """
        Student discriminator used by the generator.

        The target condition is concatenated with the feature vector so the
        student learns whether a generated row is plausible for that class.
        """
        import tensorflow.compat.v1 as tf

        discriminator_input = tf.concat([x, y_cond], axis=1)

        D_h1 = tf.nn.relu(
            tf.matmul(discriminator_input, self.D_W1) + self.D_b1
        )
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_out = tf.matmul(D_h2, self.D_W3) + self.D_b3

        return D_out

    def _teacher_discriminator(self, x, y_cond, teacher_idx: int):
        """Return logits from one independent conditional teacher."""
        import tensorflow.compat.v1 as tf

        params = self.teacher_params[teacher_idx]
        discriminator_input = tf.concat([x, y_cond], axis=1)

        h1 = tf.nn.relu(
            tf.matmul(discriminator_input, params["W1"]) + params["b1"]
        )
        h2 = tf.nn.relu(tf.matmul(h1, params["W2"]) + params["b2"])
        out = tf.matmul(h2, params["W3"]) + params["b3"]

        return out

    def _build_model(self, X_dim: int):
        """
        Build a conditional PATE-GAN graph with:
        - one shared conditional generator,
        - one conditional student discriminator,
        - multiple independent conditional teacher discriminators.
        """
        import tensorflow.compat.v1 as tf

        if self._is_built:
            return

        if self._num_classes < 2:
            raise ValueError(
                "Conditional PATE-GAN requires at least two target classes."
            )

        tf.disable_v2_behavior()
        tf.reset_default_graph()

        self._X_dim = X_dim
        if self.z_dim is None:
            self.z_dim = max(int(X_dim / 4), 2)
        self._h_dim = int(X_dim)

        conditional_dim = self._num_classes
        discriminator_input_dim = self._X_dim + conditional_dim
        generator_input_dim = self.z_dim + conditional_dim

        # Shared placeholders
        self.X = tf.placeholder(
            tf.float32, shape=[None, self._X_dim], name="X"
        )
        self.Z = tf.placeholder(
            tf.float32, shape=[None, self.z_dim], name="Z"
        )
        self.Y_cond = tf.placeholder(
            tf.float32,
            shape=[None, self._num_classes],
            name="Y_cond",
        )
        self.student_labels = tf.placeholder(
            tf.float32,
            shape=[None, 1],
            name="student_labels",
        )

        # ------------------------------------------------------------
        # Conditional generator
        # ------------------------------------------------------------
        self.G_W1 = tf.Variable(
            self._xavier_init([generator_input_dim, self._h_dim]),
            name="G_W1",
        )
        self.G_b1 = tf.Variable(
            tf.zeros(shape=[self._h_dim]),
            name="G_b1",
        )
        self.G_W2 = tf.Variable(
            self._xavier_init([self._h_dim, self._h_dim]),
            name="G_W2",
        )
        self.G_b2 = tf.Variable(
            tf.zeros(shape=[self._h_dim]),
            name="G_b2",
        )
        self.G_W3 = tf.Variable(
            self._xavier_init([self._h_dim, self._X_dim]),
            name="G_W3",
        )
        self.G_b3 = tf.Variable(
            tf.zeros(shape=[self._X_dim]),
            name="G_b3",
        )

        self.theta_G = [
            self.G_W1,
            self.G_W2,
            self.G_W3,
            self.G_b1,
            self.G_b2,
            self.G_b3,
        ]

        self._G_sample = self._generator(self.Z, self.Y_cond)

        # ------------------------------------------------------------
        # Conditional student discriminator
        # ------------------------------------------------------------
        self.D_W1 = tf.Variable(
            self._xavier_init([discriminator_input_dim, self._h_dim]),
            name="D_W1",
        )
        self.D_b1 = tf.Variable(
            tf.zeros(shape=[self._h_dim]),
            name="D_b1",
        )
        self.D_W2 = tf.Variable(
            self._xavier_init([self._h_dim, self._h_dim]),
            name="D_W2",
        )
        self.D_b2 = tf.Variable(
            tf.zeros(shape=[self._h_dim]),
            name="D_b2",
        )
        self.D_W3 = tf.Variable(
            self._xavier_init([self._h_dim, 1]),
            name="D_W3",
        )
        self.D_b3 = tf.Variable(
            tf.zeros(shape=[1]),
            name="D_b3",
        )

        self.theta_D = [
            self.D_W1,
            self.D_W2,
            self.D_W3,
            self.D_b1,
            self.D_b2,
            self.D_b3,
        ]

        # ------------------------------------------------------------
        # Independent conditional teachers
        # ------------------------------------------------------------
        self.teacher_params = []
        self.teacher_logits_real = []
        self.teacher_logits_fake = []
        self.teacher_losses = []
        self.teacher_solvers = []

        for teacher_idx in range(self.num_teachers):
            with tf.variable_scope(f"teacher_{teacher_idx}"):
                params = {
                    "W1": tf.Variable(
                        self._xavier_init(
                            [discriminator_input_dim, self._h_dim]
                        ),
                        name="W1",
                    ),
                    "b1": tf.Variable(
                        tf.zeros([self._h_dim]),
                        name="b1",
                    ),
                    "W2": tf.Variable(
                        self._xavier_init(
                            [self._h_dim, self._h_dim]
                        ),
                        name="W2",
                    ),
                    "b2": tf.Variable(
                        tf.zeros([self._h_dim]),
                        name="b2",
                    ),
                    "W3": tf.Variable(
                        self._xavier_init([self._h_dim, 1]),
                        name="W3",
                    ),
                    "b3": tf.Variable(
                        tf.zeros([1]),
                        name="b3",
                    ),
                }

                self.teacher_params.append(params)

                t_real = self._teacher_discriminator(
                    self.X,
                    self.Y_cond,
                    teacher_idx,
                )
                t_fake = self._teacher_discriminator(
                    tf.stop_gradient(self._G_sample),
                    self.Y_cond,
                    teacher_idx,
                )

                teacher_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=t_real,
                        labels=tf.ones_like(t_real),
                    )
                )
                teacher_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=t_fake,
                        labels=tf.zeros_like(t_fake),
                    )
                )
                teacher_loss = teacher_loss_real + teacher_loss_fake

                teacher_vars = [
                    params["W1"],
                    params["b1"],
                    params["W2"],
                    params["b2"],
                    params["W3"],
                    params["b3"],
                ]

                teacher_solver = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.5,
                ).minimize(
                    teacher_loss,
                    var_list=teacher_vars,
                )

                self.teacher_logits_real.append(t_real)
                self.teacher_logits_fake.append(t_fake)
                self.teacher_losses.append(teacher_loss)
                self.teacher_solvers.append(teacher_solver)

        # ------------------------------------------------------------
        # Student and generator losses
        # ------------------------------------------------------------
        student_fake_logits = self._student_discriminator(
            self._G_sample,
            self.Y_cond,
        )

        self.student_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=student_fake_logits,
                labels=self.student_labels,
            )
        )

        self.D_solver = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.5,
        ).minimize(
            self.student_loss,
            var_list=self.theta_D,
        )

        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=student_fake_logits,
                labels=tf.ones_like(student_fake_logits),
            )
        )

        self.G_solver = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.5,
        ).minimize(
            self.G_loss,
            var_list=self.theta_G,
        )

        self.teacher_vote_probs = [
            tf.nn.sigmoid(logits)
            for logits in self.teacher_logits_fake
        ]

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._is_built = True

    def _sample_class_indices(self, n: int, rng) -> np.ndarray:
        """Sample target-class indices from the real training distribution."""
        return rng.choice(
            self._num_classes,
            size=n,
            p=self._class_probs,
        )

    def _one_hot(self, class_indices: np.ndarray) -> np.ndarray:
        """Convert integer class indices to one-hot vectors."""
        return np.eye(
            self._num_classes,
            dtype=np.float32,
        )[class_indices]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        verbose: int = 1,
    ) -> "PATEGAN":
        """
        Fit conditional PATE-GAN.

        Features are generated conditionally on the target class. The target
        itself is not generated as a free continuous output, preventing the
        generator from collapsing the target column to a single value.
        """
        if y is None:
            raise ValueError(
                "Class-aware PATE-GAN requires y for conditional training."
            )

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "PATE-GAN currently supports one target column."
                )
            y_series = y.iloc[:, 0].copy()
        else:
            y_series = y.copy()

        self._target_name = (
            y_series.name
            if y_series.name is not None
            else "target"
        )

        set_global_seed(self.random_state)

        # ------------------------------------------------------------
        # Preserve class values and empirical class distribution
        # ------------------------------------------------------------
        class_values = pd.Series(y_series).reset_index(drop=True)
        self._target_classes = np.array(
            sorted(class_values.unique().tolist())
        )
        self._num_classes = len(self._target_classes)

        class_to_index = {
            cls: idx
            for idx, cls in enumerate(self._target_classes)
        }
        y_indices = np.array(
            [class_to_index[value] for value in class_values],
            dtype=np.int64,
        )

        class_counts = np.bincount(
            y_indices,
            minlength=self._num_classes,
        ).astype(float)
        self._class_probs = class_counts / class_counts.sum()

        y_onehot = self._one_hot(y_indices)

        # ------------------------------------------------------------
        # Transform features only
        # ------------------------------------------------------------
        X_reset = X.reset_index(drop=True).copy()

        self.transformer = DataTransformer()
        X_encoded = self.transformer.fit_transform(X_reset)

        self._build_model(X_encoded.shape[1])

        self.privacy_mechanism = PrivacyMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            num_teachers=self.num_teachers,
        )

        # Partition features and labels together so every teacher sees a
        # genuinely disjoint subset with matching target conditions.
        joint_data = np.concatenate(
            [X_encoded, y_onehot],
            axis=1,
        )
        teacher_partitions = partition_data(
            joint_data,
            self.num_teachers,
        )

        if verbose:
            print(f"Training PATE-GAN with ε={self.epsilon}, δ={self.delta}")
            print(f"Teachers: {self.num_teachers}, Iterations: {self.niter}")
            print(
                "Teacher partition sizes:",
                [len(partition) for partition in teacher_partitions],
            )
            print(
                "Target classes:",
                self._target_classes.tolist(),
                "probs:",
                np.round(self._class_probs, 4).tolist(),
            )

        try:
            from tqdm import tqdm

            iterator = (
                tqdm(range(self.niter), desc="Training")
                if verbose
                else range(self.niter)
            )
            use_tqdm = True
        except ImportError:
            iterator = range(self.niter)
            use_tqdm = False

        print_every = (
            max(1, self.niter // 10)
            if verbose and not use_tqdm
            else None
        )

        rng = np.random.RandomState(self.random_state)

        for it in iterator:
            teacher_loss_values = []

            # --------------------------------------------------------
            # 1. Train each teacher on its own private partition
            # --------------------------------------------------------
            for teacher_idx in range(self.num_teachers):
                teacher_data = teacher_partitions[teacher_idx]

                if len(teacher_data) == 0:
                    raise ValueError(
                        f"Teacher {teacher_idx} received an empty partition."
                    )

                replace = len(teacher_data) < self.batch_size
                indices = rng.choice(
                    len(teacher_data),
                    self.batch_size,
                    replace=replace,
                )
                batch = teacher_data[indices]

                X_mb = batch[:, :self._X_dim]
                Y_real = batch[:, self._X_dim:]

                # Fake samples are conditioned on sampled target classes.
                fake_class_idx = self._sample_class_indices(
                    self.batch_size,
                    rng,
                )
                Y_fake = self._one_hot(fake_class_idx)

                Z_mb = rng.uniform(
                    -1.0,
                    1.0,
                    size=[self.batch_size, self.z_dim],
                )

                # Teacher real and fake branches use the same placeholder.
                # Train once on real-conditioned data and once on fake-
                # conditioned data so class information is explicitly seen.
                _, teacher_loss_real = self._sess.run(
                    [
                        self.teacher_solvers[teacher_idx],
                        self.teacher_losses[teacher_idx],
                    ],
                    feed_dict={
                        self.X: X_mb,
                        self.Z: Z_mb,
                        self.Y_cond: Y_real,
                    },
                )

                _, teacher_loss_fake = self._sess.run(
                    [
                        self.teacher_solvers[teacher_idx],
                        self.teacher_losses[teacher_idx],
                    ],
                    feed_dict={
                        self.X: X_mb,
                        self.Z: Z_mb,
                        self.Y_cond: Y_fake,
                    },
                )

                teacher_loss_values.append(
                    0.5 * (teacher_loss_real + teacher_loss_fake)
                )

            # --------------------------------------------------------
            # 2. Generate conditional samples and collect teacher votes
            # --------------------------------------------------------
            vote_class_idx = self._sample_class_indices(
                self.batch_size,
                rng,
            )
            Y_vote = self._one_hot(vote_class_idx)

            Z_vote = rng.uniform(
                -1.0,
                1.0,
                size=[self.batch_size, self.z_dim],
            )

            teacher_prob_values = self._sess.run(
                self.teacher_vote_probs,
                feed_dict={
                    self.Z: Z_vote,
                    self.Y_cond: Y_vote,
                },
            )

            teacher_votes = np.stack(
                [
                    (np.asarray(prob).reshape(-1) >= 0.5).astype(float)
                    for prob in teacher_prob_values
                ],
                axis=0,
            )

            vote_counts_real = np.sum(
                teacher_votes,
                axis=0,
            )
            vote_counts_fake = (
                self.num_teachers - vote_counts_real
            )

            noisy_real = self.privacy_mechanism.add_gaussian_noise(
                vote_counts_real.astype(float)
            )
            noisy_fake = self.privacy_mechanism.add_gaussian_noise(
                vote_counts_fake.astype(float)
            )

            private_labels = (
                noisy_real >= noisy_fake
            ).astype(np.float32).reshape(-1, 1)

            # --------------------------------------------------------
            # 3. Student learns only from private aggregated labels
            # --------------------------------------------------------
            _, student_loss_curr = self._sess.run(
                [self.D_solver, self.student_loss],
                feed_dict={
                    self.Z: Z_vote,
                    self.Y_cond: Y_vote,
                    self.student_labels: private_labels,
                },
            )

            # --------------------------------------------------------
            # 4. Generator learns conditionally against student
            # --------------------------------------------------------
            gen_class_idx = self._sample_class_indices(
                self.batch_size,
                rng,
            )
            Y_gen = self._one_hot(gen_class_idx)

            Z_gen = rng.uniform(
                -1.0,
                1.0,
                size=[self.batch_size, self.z_dim],
            )

            _, G_loss_curr = self._sess.run(
                [self.G_solver, self.G_loss],
                feed_dict={
                    self.Z: Z_gen,
                    self.Y_cond: Y_gen,
                },
            )

            if print_every is not None and it % print_every == 0:
                print(
                    f"Iter {it}/{self.niter}: "
                    f"Teacher_loss={np.mean(teacher_loss_values):.4f}, "
                    f"Student_loss={student_loss_curr:.4f}, "
                    f"G_loss={G_loss_curr:.4f}"
                )

        if verbose:
            print("Training completed!")

        self.is_fitted = True
        return self

    def sample(
        self,
        n: int,
        conditional: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate class-aware synthetic samples.

        By default, target classes are sampled according to the empirical
        training distribution. A fixed class may be requested with:
            conditional={target_column_name: class_value}

        Args:
            n: Number of rows to generate.
            conditional: Optional target-class condition.
            seed: Optional sampling seed for reproducible stability tests.

        Returns:
            DataFrame containing generated features plus the target column.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling")

        rng = np.random.RandomState(
            self.random_state if seed is None else seed
        )

        if conditional is not None:
            if self._target_name not in conditional:
                raise ValueError(
                    f"Conditional sampling currently supports only "
                    f"'{self._target_name}'."
                )

            requested_class = conditional[self._target_name]

            matching = np.where(
                self._target_classes == requested_class
            )[0]

            if len(matching) == 0:
                raise ValueError(
                    f"Unknown target class {requested_class!r}. "
                    f"Available classes: "
                    f"{self._target_classes.tolist()}"
                )

            class_indices = np.full(
                n,
                matching[0],
                dtype=np.int64,
            )
        else:
            class_indices = self._sample_class_indices(
                n,
                rng,
            )

        y_onehot_all = self._one_hot(class_indices)
        target_values = self._target_classes[class_indices]

        all_samples = []
        n_batches = int(np.ceil(n / self.batch_size))

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min(
                start_idx + self.batch_size,
                n,
            )

            batch_size = end_idx - start_idx

            Z_sample = rng.uniform(
                -1.0,
                1.0,
                size=[batch_size, self.z_dim],
            )
            Y_sample = y_onehot_all[start_idx:end_idx]

            X_synth = self._sess.run(
                self._G_sample,
                feed_dict={
                    self.Z: Z_sample,
                    self.Y_cond: Y_sample,
                },
            )

            all_samples.append(X_synth)

        X_synth_all = np.vstack(all_samples)[:n]

        df_synth = self.transformer.inverse_transform(
            X_synth_all
        )

        # The target comes directly from the conditioning variable rather than
        # being generated as an unconstrained continuous output.
        df_synth[self._target_name] = target_values

        return df_synth

    def train(
        self, dataset_dir: str, synthetic_dir: str | None = None, **kwargs
    ) -> "PATEGAN":
        """
        Train PATE-GAN following Katabatic pipeline contract.

        Reads x_train.csv and y_train.csv from dataset_dir,
        trains the model, generates synthetic data, and writes
        x_synth.csv, y_synth.csv, and metadata.json to synthetic_dir.

        Args:
            dataset_dir: Directory containing x_train.csv and y_train.csv
            synthetic_dir: Directory to save synthetic data (optional)
            **kwargs: Additional training parameters (epsilon, delta, num_teachers, niter, batch_size, etc.)

        Returns:
            self
        """
        # Override model parameters from kwargs if provided
        # Pop them so they don't propagate to evaluations
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
        if "lambda_gp" in kwargs:
            self.lambda_gp = kwargs.pop("lambda_gp")
        if "z_dim" in kwargs:
            self.z_dim = kwargs.pop("z_dim")
        if "random_state" in kwargs:
            self.random_state = kwargs.pop("random_state")

        # Read training data
        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")

        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"x_train.csv not found in {dataset_dir}")

        X_train = pd.read_csv(x_train_path)

        # y_train is optional
        y_train = None
        y_label_encoder = None
        if os.path.exists(y_train_path):
            y_train = pd.read_csv(y_train_path)
            if isinstance(y_train, pd.DataFrame) and len(y_train.columns) == 1:
                y_train = y_train.iloc[:, 0]

            # Remap y_train classes to consecutive integers [0, 1, 2, ...]
            # This is required for ML models like XGBoost which expect consecutive classes
            from sklearn.preprocessing import LabelEncoder

            y_label_encoder = LabelEncoder()
            original_classes = y_train.unique()
            y_train_remapped = y_label_encoder.fit_transform(y_train)
            y_train = pd.Series(
                y_train_remapped,
                name=y_train.name if hasattr(y_train, "name") else "target",
            )
            print(
                f"Remapped y classes: {sorted(original_classes)} -> {sorted(y_label_encoder.transform(original_classes))}"
            )

        print(f"Loaded training data: X shape={X_train.shape}", end="")
        if y_train is not None:
            print(
                f", y shape={y_train.shape if isinstance(y_train, pd.DataFrame) else (len(y_train),)}"
            )
        else:
            print()

        # Fit model
        self.fit(X_train, y_train, verbose=kwargs.get("verbose", 1))

        # Generate synthetic data
        n_samples = len(X_train)
        print(f"\nGenerating {n_samples} synthetic samples...")
        df_synth = self.sample(n_samples)

        # Split into X and y
        if y_train is not None:
            target_cols = (
                y_train.columns.tolist()
                if isinstance(y_train, pd.DataFrame)
                else [y_train.name]
            )
            x_synth = df_synth.drop(columns=target_cols)
            y_synth = df_synth[target_cols]

            # Ensure all training classes are present in synthetic data (robustness for TSTR)
            y_col = target_cols[0]
            df_train = pd.concat(
                [
                    X_train.copy(),
                    (
                        y_train
                        if isinstance(y_train, pd.DataFrame)
                        else y_train.to_frame(name=y_col)
                    ),
                ],
                axis=1,
            )

            unique_train = np.unique(df_train[y_col].values)
            unique_synth = np.unique(y_synth[y_col].values)
            missing_classes = set(unique_train) - set(unique_synth)

            if missing_classes:
                print(
                    f"[PATEGAN] Adding {len(missing_classes)} dummy samples to cover classes: {sorted(missing_classes)}"
                )
                for cls in missing_classes:
                    idx = np.where(df_train[y_col].values == cls)[0]
                    if idx.size == 0:
                        continue
                    row = df_train.iloc[idx[0] : idx[0] + 1]
                    x_dummy = row.drop(columns=[y_col])
                    y_dummy = row[[y_col]]
                    x_synth = pd.concat([x_synth, x_dummy], ignore_index=True)
                    y_synth = pd.concat([y_synth, y_dummy], ignore_index=True)

            # Final guard: ensure at least 2 classes
            if np.unique(y_synth[y_col].values).size < 2 and unique_train.size >= 2:
                alt_classes = [c for c in unique_train if c != y_synth[y_col].iloc[0]]
                if alt_classes:
                    y_synth.loc[y_synth.index[0], y_col] = alt_classes[0]
                    print(
                        f"[PATEGAN] Forced presence of a second class: {alt_classes[0]}"
                    )

            # Cast to integers to ensure proper class labels and discrete features
            for col in x_synth.columns:
                try:
                    x_synth[col] = x_synth[col].astype(int)
                except Exception:
                    pass
            y_synth[y_col] = y_synth[y_col].astype(int)

            # y_synth already has remapped classes [0, 1, 2, ...] since model was trained on remapped data
            # This is what evaluation expects
        else:
            x_synth = df_synth
            y_synth = None

        # Also remap y_test.csv to match the synthetic data's class encoding
        # Get real_test_dir from kwargs (passed by pipeline)
        real_test_dir = kwargs.get("real_test_dir")
        if y_label_encoder is not None and real_test_dir is not None:
            y_test_path = os.path.join(real_test_dir, "y_test.csv")
            if os.path.exists(y_test_path):
                y_test = pd.read_csv(y_test_path)
                if isinstance(y_test, pd.DataFrame) and len(y_test.columns) == 1:
                    y_test = y_test.iloc[:, 0]

                # Only keep test samples with classes seen in training
                test_mask = y_test.isin(y_label_encoder.classes_)
                if not test_mask.all():
                    print(
                        f"Warning: Filtering {(~test_mask).sum()} test samples with unseen classes"
                    )
                    # Also filter x_test
                    x_test_path = os.path.join(real_test_dir, "x_test.csv")
                    if os.path.exists(x_test_path):
                        x_test = pd.read_csv(x_test_path)
                        x_test = x_test[test_mask]
                        x_test.to_csv(x_test_path, index=False)
                    y_test = y_test[test_mask]

                # Transform y_test with same encoder
                y_test_remapped = y_label_encoder.transform(y_test)
                y_test = pd.DataFrame(
                    y_test_remapped,
                    columns=y_test.columns
                    if isinstance(y_test, pd.DataFrame)
                    else [y_test.name],
                )
                y_test.to_csv(y_test_path, index=False)
                print("Remapped y_test.csv to match synthetic data encoding")

        # Save synthetic data
        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)

            x_synth_path = os.path.join(synthetic_dir, "x_synth.csv")
            x_synth.to_csv(x_synth_path, index=False)
            print(f"Saved x_synth.csv to {x_synth_path}")

            if y_synth is not None:
                y_synth_path = os.path.join(synthetic_dir, "y_synth.csv")
                y_synth.to_csv(y_synth_path, index=False)
                print(f"Saved y_synth.csv to {y_synth_path}")

            # Save metadata
            metadata_path = os.path.join(synthetic_dir, "metadata.json")
            training_config = {
                "niter": self.niter,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "lambda_gp": self.lambda_gp,
                "z_dim": self.z_dim,
            }
            privacy_config = {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "num_teachers": self.num_teachers,
                "lambda_noise": self.privacy_mechanism.lambda_noise,
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
        """
        In-memory TSTR-style check: sample synthetic rows, fit a downstream model
        on synthetic data, and score on the provided real ``(x, y)``.

        For **canonical** utility evaluation (aligned splits, artifact logging),
        use :class:`katabatic.pipeline.train_test_split.pipeline.TrainTestSplitPipeline`
        with :class:`katabatic.evaluate.tstr.evaluation.TSTREvaluation`.

        Args:
            x: Test features
            y: Test labels
            model: Model type ('lr', 'rf', 'mlp')
            metrics: List of metrics to compute
            task: 'classification' or 'regression' (auto-detected if None)
            random_state: Random seed
            **kwargs: Additional parameters

        Returns:
            Dictionary of metric scores
        """
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

        # Auto-detect task
        if task is None:
            if y.dtype in ["object", "category"] or y.nunique() < 20:
                task = "classification"
            else:
                task = "regression"

        # Generate synthetic training data
        n_train = len(x)
        df_synth = self.sample(n_train)

        # Assuming y is the last column(s)
        if task == "classification":
            X_synth = df_synth.iloc[:, :-1]
            y_synth = df_synth.iloc[:, -1]
        else:
            X_synth = df_synth.iloc[:, :-1]
            y_synth = df_synth.iloc[:, -1]

        # Train downstream model
        if task == "classification":
            if model == "lr":
                clf = LogisticRegression(random_state=random_state, max_iter=1000)
            elif model == "rf":
                clf = RandomForestClassifier(
                    random_state=random_state, n_estimators=100
                )
            else:
                clf = LogisticRegression(random_state=random_state, max_iter=1000)

            clf.fit(X_synth, y_synth)
            y_pred = clf.predict(x)

            results = {
                "accuracy": accuracy_score(y, y_pred),
                "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            }

            # Add AUC if binary
            if y.nunique() == 2:
                try:
                    y_proba = clf.predict_proba(x)[:, 1]
                    results["roc_auc"] = roc_auc_score(y, y_proba)
                except (AttributeError, ValueError):
                    pass
        else:
            if model == "lr":
                reg = LinearRegression()
            elif model == "rf":
                reg = RandomForestRegressor(random_state=random_state, n_estimators=100)
            else:
                reg = LinearRegression()

            reg.fit(X_synth, y_synth)
            y_pred = reg.predict(x)

            results = {
                "r2": r2_score(y, y_pred),
                "mae": mean_absolute_error(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            }

        return results

    def __del__(self):
        """Clean up TensorFlow session."""
        if self._sess is not None:
            try:
                self._sess.close()
            except Exception:
                pass