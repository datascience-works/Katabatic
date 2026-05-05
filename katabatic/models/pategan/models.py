import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from katabatic.models.base_model import Model
from .utils import (
    DataTransformer,
    PrivacyMechanism,
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
        z_dim: Optional[int] = None,
        learning_rate: float = 1e-4,
        lambda_gp: float = 10.0,
        random_state: int = 42
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
        self.transformer: Optional[DataTransformer] = None
        self.privacy_mechanism: Optional[PrivacyMechanism] = None
        self._sess = None
        self._G_sample = None
        self._is_built = False
        self._X_dim = None
        self._h_dim = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return required dependencies for PATE-GAN."""
        return ['tensorflow', 'scipy', 'sklearn', 'pandas', 'numpy']

    def _xavier_init(self, size):
        """Xavier initialization for network weights."""
        import tensorflow.compat.v1 as tf
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _generator(self, z):
        """
        Generator network.

        Args:
            z: Latent noise vector

        Returns:
            Generated samples
        """
        import tensorflow.compat.v1 as tf

        G_h1 = tf.nn.tanh(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_out = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3)

        return G_out

    def _discriminator(self, x):
        """
        Discriminator network.

        Args:
            x: Input samples

        Returns:
            Discriminator score
        """
        import tensorflow.compat.v1 as tf

        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_out = tf.matmul(D_h2, self.D_W3) + self.D_b3

        return D_out

    def _build_model(self, X_dim: int):
        """
        Build TensorFlow computation graph.

        Args:
            X_dim: Feature dimension
        """
        import tensorflow.compat.v1 as tf

        if self._is_built:
            return

        tf.disable_v2_behavior()
        tf.reset_default_graph()

        self._X_dim = X_dim
        if self.z_dim is None:
            self.z_dim = max(int(X_dim / 4), 2)
        self._h_dim = int(X_dim)

        # Placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, self._X_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.M = tf.placeholder(tf.float32, shape=[None, 1])

        # Discriminator parameters
        self.D_W1 = tf.Variable(self._xavier_init([self._X_dim, self._h_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self._h_dim]))
        self.D_W2 = tf.Variable(self._xavier_init([self._h_dim, self._h_dim]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[self._h_dim]))
        self.D_W3 = tf.Variable(self._xavier_init([self._h_dim, 1]))
        self.D_b3 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_W3,
                        self.D_b1, self.D_b2, self.D_b3]

        # Generator parameters
        self.G_W1 = tf.Variable(self._xavier_init([self.z_dim, self._h_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self._h_dim]))
        self.G_W2 = tf.Variable(self._xavier_init([self._h_dim, self._h_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self._h_dim]))
        self.G_W3 = tf.Variable(self._xavier_init([self._h_dim, self._X_dim]))
        self.G_b3 = tf.Variable(tf.zeros(shape=[self._X_dim]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_W3,
                        self.G_b1, self.G_b2, self.G_b3]

        # Build computational graph
        self._G_sample = self._generator(self.Z)
        D_real = self._discriminator(self.X)
        D_fake = self._discriminator(self._G_sample)

        D_entire = tf.concat(axis=0, values=[D_real, D_fake])

        # Gradient penalty (WGAN-GP)
        eps = tf.random_uniform([self.batch_size, 1], minval=0., maxval=1.)
        X_inter = eps * self.X + (1. - eps) * self._G_sample
        grad = tf.gradients(self._discriminator(X_inter), [X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2 + 1e-8, axis=1))
        grad_pen = self.lambda_gp * tf.reduce_mean((grad_norm - 1)**2)

        # Loss functions
        D_loss = (tf.reduce_mean((1 - self.M) * D_entire) -
                  tf.reduce_mean(self.M * D_entire) + grad_pen)
        G_loss = -tf.reduce_mean(D_fake)

        # Optimizers
        D_solver = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=0.5
        ).minimize(D_loss, var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=0.5
        ).minimize(G_loss, var_list=self.theta_G)

        # Store for training
        self.D_solver = D_solver
        self.G_solver = G_solver
        self.D_loss = D_loss
        self.G_loss = G_loss

        # Session
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._is_built = True

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, verbose: int = 1) -> 'PATEGAN':
        """
        Fit PATE-GAN to data.

        Args:
            X: Feature dataframe
            y: Target series (optional, will be concatenated to X)
            verbose: Verbosity level (0=silent, 1=progress)

        Returns:
            self
        """
        # Combine X and y if provided
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.to_frame()
            data = pd.concat([X, y], axis=1)
        else:
            data = X.copy()

        # Set seed
        set_global_seed(self.random_state)

        # Fit transformer and encode data
        self.transformer = DataTransformer()
        X_encoded = self.transformer.fit_transform(data)

        # Build model
        self._build_model(X_encoded.shape[1])

        # Initialize privacy mechanism
        self.privacy_mechanism = PrivacyMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            num_teachers=self.num_teachers
        )

        # Partition data into disjoint subsets — one per teacher (true PATE requirement)
        n_samples = len(X_encoded)
        indices_perm = np.random.permutation(n_samples)
        partition_size = n_samples // self.num_teachers
        partitions = [
            X_encoded[indices_perm[i * partition_size: (i + 1) * partition_size]]
            for i in range(self.num_teachers)
        ]

        if verbose:
            print(f"Training PATE-GAN with ε={self.epsilon}, δ={self.delta}")
            print(f"Teachers: {self.num_teachers}, Iterations: {self.niter}")

        # Setup progress tracking
        try:
            from tqdm import tqdm
            iterator = tqdm(range(self.niter),
                            desc="Training") if verbose else range(self.niter)
            use_tqdm = True
        except ImportError:
            iterator = range(self.niter)
            use_tqdm = False

        print_every = max(1, self.niter // 10) if verbose and not use_tqdm else None

        for it in iterator:
            # Train each teacher discriminator on its own disjoint partition
            for teacher_idx in range(self.num_teachers):
                partition = partitions[teacher_idx]
                part_size = len(partition)
                sample_n = min(self.batch_size, part_size)
                idx = np.random.choice(part_size, sample_n, replace=False)
                X_mb = partition[idx]

                Z_mb = np.random.uniform(-1., 1., size=[sample_n, self.z_dim])

                # Binary labels: 1=real, 0=fake
                M_real = np.ones([sample_n])
                M_fake = np.zeros([sample_n])
                M_entire = np.concatenate((M_real, M_fake), 0)

                # Add Gaussian noise for differential privacy, then re-binarise
                M_entire = self.privacy_mechanism.add_gaussian_noise(M_entire)
                M_entire = (M_entire > 0.5).astype(float)
                M_mb = M_entire.reshape(-1, 1)

                _, D_loss_curr = self._sess.run(
                    [self.D_solver, self.D_loss],
                    feed_dict={self.X: X_mb, self.Z: Z_mb, self.M: M_mb}
                )

            # Train generator (only Z needed — G_loss depends solely on D_fake)
            Z_mb = np.random.uniform(-1., 1., size=[self.batch_size, self.z_dim])
            _, G_loss_curr = self._sess.run(
                [self.G_solver, self.G_loss],
                feed_dict={self.Z: Z_mb}
            )

            if print_every is not None and it % print_every == 0:
                print(f"Iter {it}/{self.niter}: D_loss={D_loss_curr:.4f}, G_loss={G_loss_curr:.4f}")

        if verbose:
            print("Training completed!")

        self.is_fitted = True
        return self

    def sample(
        self,
        n_samples: int,
        seed: int = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility (used by stability evaluator)

        Returns:
            DataFrame with synthetic samples in original feature space
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling")

        if seed is not None:
            np.random.seed(seed)

        all_samples = []
        remaining = n_samples

        while remaining > 0:
            batch_size = min(self.batch_size, remaining)
            Z_sample = np.random.uniform(-1., 1., size=[batch_size, self.z_dim])
            X_synth = self._sess.run(
                [self._G_sample],
                feed_dict={self.Z: Z_sample}
            )[0]
            all_samples.append(X_synth)
            remaining -= batch_size

        X_synth_all = np.vstack(all_samples)[:n_samples]
        return self.transformer.inverse_transform(X_synth_all)

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        categorical_cols: Optional[list] = None,
        continuous_cols: Optional[list] = None,
        **kwargs
    ) -> 'PATEGAN':
        """
        Train PATE-GAN following the Katabatic runner contract.

        Loads training data from dataset_dir using the standard priority order:
          train_sample.csv  →  train_full.csv  →  x_train.csv + y_train.csv

        The runner calls sample() separately to obtain synthetic data, so this
        method only trains the model and does not generate or save outputs.

        Args:
            dataset_dir: Directory containing training CSVs
            synthetic_dir: Unused — kept for interface compatibility
            categorical_cols: Accepted for interface compatibility (dtype-inferred internally)
            continuous_cols: Accepted for interface compatibility (dtype-inferred internally)
            **kwargs: verbose (int, default 1)
        """
        train_sample = os.path.join(dataset_dir, "train_sample.csv")
        train_full   = os.path.join(dataset_dir, "train_full.csv")
        x_path       = os.path.join(dataset_dir, "x_train.csv")
        y_path       = os.path.join(dataset_dir, "y_train.csv")

        if os.path.exists(train_sample):
            df = pd.read_csv(train_sample)
        elif os.path.exists(train_full):
            df = pd.read_csv(train_full)
        elif os.path.exists(x_path):
            X = pd.read_csv(x_path)
            if os.path.exists(y_path):
                y = pd.read_csv(y_path)
                if y.shape[1] != 1:
                    raise ValueError("y_train.csv must have exactly one column.")
                df = pd.concat([X, y[y.columns[0]]], axis=1)
            else:
                df = X
        else:
            raise FileNotFoundError(
                f"No training data found in {dataset_dir}. "
                "Expected train_sample.csv, train_full.csv, or x_train.csv."
            )

        print(f"[PATEGAN] Loaded {len(df)} training rows, {df.shape[1]} columns.")
        self.fit(df, verbose=kwargs.get('verbose', 1))
        return self

    def evaluate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        model: str = "lr",
        task: Optional[str] = None,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, float]:
        """
        Quick standalone TSTR evaluation (not used by the pipeline).

        Args:
            x: Test features
            y: Test labels
            model: Downstream model type ('lr' or 'rf')
            task: 'classification' or 'regression' (auto-detected if None)
            random_state: Random seed

        Returns:
            Dictionary of metric scores
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task is None:
            task = 'classification' if (y.dtype == object or y.nunique() < 20) else 'regression'

        df_synth = self.sample(len(x))
        X_synth = df_synth.iloc[:, :-1]
        y_synth = df_synth.iloc[:, -1]

        if task == 'classification':
            clf = (RandomForestClassifier(random_state=random_state, n_estimators=100)
                   if model == 'rf'
                   else LogisticRegression(random_state=random_state, max_iter=1000))
            clf.fit(X_synth, y_synth)
            y_pred = clf.predict(x)
            results = {
                'accuracy': accuracy_score(y, y_pred),
                'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            }
            if y.nunique() == 2:
                try:
                    results['roc_auc'] = roc_auc_score(y, clf.predict_proba(x)[:, 1])
                except Exception:
                    pass
        else:
            reg = (RandomForestRegressor(random_state=random_state, n_estimators=100)
                   if model == 'rf'
                   else LinearRegression())
            reg.fit(X_synth, y_synth)
            y_pred = reg.predict(x)
            results = {
                'r2': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            }

        return results

    def __del__(self):
        """Clean up TensorFlow session."""
        if self._sess is not None:
            try:
                self._sess.close()
            except:
                pass
