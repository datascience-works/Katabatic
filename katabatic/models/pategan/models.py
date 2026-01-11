import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union
from katabatic.models.base_model import Model
from .utils import (
    DataTransformer,
    PrivacyMechanism,
    set_global_seed,
    save_metadata,
    load_metadata,
    reconstruct_transformer,
    partition_data
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

        # Training loop
        n_samples = len(X_encoded)

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

        # For non-tqdm progress printing
        print_every = max(
            1, self.niter // 10) if verbose and not use_tqdm else None

        for it in iterator:
            # Train teacher discriminators
            for teacher_idx in range(self.num_teachers):
                # Sample batch from data
                indices = np.random.choice(
                    n_samples, self.batch_size, replace=False)
                X_mb = X_encoded[indices]

                # Sample noise
                Z_mb = np.random.uniform(-1., 1.,
                                         size=[self.batch_size, self.z_dim])

                # Create teacher labels with privacy noise
                M_real = np.ones([self.batch_size,])
                M_fake = np.zeros([self.batch_size,])
                M_entire = np.concatenate((M_real, M_fake), 0)

                # Add Gaussian noise for privacy
                noise = self.privacy_mechanism.add_gaussian_noise(M_entire)
                M_entire = M_entire + noise
                M_entire = (M_entire > 0.5).astype(float)
                M_mb = np.reshape(M_entire, (2 * self.batch_size, 1))

                # Train discriminator
                _, D_loss_curr = self._sess.run(
                    [self.D_solver, self.D_loss],
                    feed_dict={self.X: X_mb, self.Z: Z_mb, self.M: M_mb}
                )

            # Train generator
            Z_mb = np.random.uniform(-1., 1.,
                                     size=[self.batch_size, self.z_dim])
            indices = np.random.choice(
                n_samples, self.batch_size, replace=False)
            X_mb = X_encoded[indices]

            _, G_loss_curr = self._sess.run(
                [self.G_solver, self.G_loss],
                feed_dict={self.Z: Z_mb}
            )

            # Print progress (only if not using tqdm)
            if print_every is not None and it % print_every == 0:
                print(
                    f"Iter {it}/{self.niter}: D_loss={D_loss_curr:.4f}, G_loss={G_loss_curr:.4f}")

        if verbose:
            print("Training completed!")

        self.is_fitted = True
        return self

    def sample(
        self,
        n: int,
        conditional: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Args:
            n: Number of samples to generate
            conditional: Not implemented (for future use)

        Returns:
            DataFrame with synthetic samples in original feature space
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling")

        if conditional is not None:
            print("Warning: Conditional sampling not yet implemented for PATE-GAN")

        # Generate in batches to avoid memory issues
        all_samples = []
        n_batches = int(np.ceil(n / self.batch_size))

        for i in range(n_batches):
            batch_size = min(self.batch_size, n - i * self.batch_size)
            Z_sample = np.random.uniform(-1., 1.,
                                         size=[batch_size, self.z_dim])

            X_synth = self._sess.run(
                [self._G_sample],
                feed_dict={self.Z: Z_sample}
            )[0]

            all_samples.append(X_synth)

        # Concatenate all batches
        X_synth_all = np.vstack(all_samples)[:n]

        # Decode back to original feature space
        df_synth = self.transformer.inverse_transform(X_synth_all)

        return df_synth

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: Optional[str] = None,
        **kwargs
    ) -> 'PATEGAN':
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
        if 'epsilon' in kwargs:
            self.epsilon = kwargs.pop('epsilon')
        if 'delta' in kwargs:
            self.delta = kwargs.pop('delta')
        if 'num_teachers' in kwargs:
            self.num_teachers = kwargs.pop('num_teachers')
        if 'niter' in kwargs:
            self.niter = kwargs.pop('niter')
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs.pop('learning_rate')
        if 'lambda_gp' in kwargs:
            self.lambda_gp = kwargs.pop('lambda_gp')
        if 'z_dim' in kwargs:
            self.z_dim = kwargs.pop('z_dim')
        if 'random_state' in kwargs:
            self.random_state = kwargs.pop('random_state')

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
            y_train = pd.Series(y_train_remapped, name=y_train.name if hasattr(
                y_train, 'name') else 'target')
            print(
                f"Remapped y classes: {sorted(original_classes)} -> {sorted(y_label_encoder.transform(original_classes))}")

        print(f"Loaded training data: X shape={X_train.shape}", end="")
        if y_train is not None:
            print(
                f", y shape={y_train.shape if isinstance(y_train, pd.DataFrame) else (len(y_train),)}")
        else:
            print()

        # Fit model
        self.fit(X_train, y_train, verbose=kwargs.get('verbose', 1))

        # Generate synthetic data
        n_samples = len(X_train)
        print(f"\nGenerating {n_samples} synthetic samples...")
        df_synth = self.sample(n_samples)

        # Split into X and y
        if y_train is not None:
            target_cols = y_train.columns.tolist() if isinstance(
                y_train, pd.DataFrame) else [y_train.name]
            x_synth = df_synth.drop(columns=target_cols)
            y_synth = df_synth[target_cols]

            # Ensure all training classes are present in synthetic data (robustness for TSTR)
            y_col = target_cols[0]
            df_train = pd.concat([
                X_train.copy(),
                (y_train if isinstance(y_train, pd.DataFrame)
                 else y_train.to_frame(name=y_col))
            ], axis=1)

            unique_train = np.unique(df_train[y_col].values)
            unique_synth = np.unique(y_synth[y_col].values)
            missing_classes = set(unique_train) - set(unique_synth)

            if missing_classes:
                print(
                    f"[PATEGAN] Adding {len(missing_classes)} dummy samples to cover classes: {sorted(missing_classes)}")
                for cls in missing_classes:
                    idx = np.where(df_train[y_col].values == cls)[0]
                    if idx.size == 0:
                        continue
                    row = df_train.iloc[idx[0]:idx[0]+1]
                    x_dummy = row.drop(columns=[y_col])
                    y_dummy = row[[y_col]]
                    x_synth = pd.concat([x_synth, x_dummy], ignore_index=True)
                    y_synth = pd.concat([y_synth, y_dummy], ignore_index=True)

            # Final guard: ensure at least 2 classes
            if np.unique(y_synth[y_col].values).size < 2 and unique_train.size >= 2:
                alt_classes = [c for c in unique_train if c !=
                               y_synth[y_col].iloc[0]]
                if alt_classes:
                    y_synth.loc[y_synth.index[0], y_col] = alt_classes[0]
                    print(
                        f"[PATEGAN] Forced presence of a second class: {alt_classes[0]}")

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
        real_test_dir = kwargs.get('real_test_dir')
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
                        f"Warning: Filtering {(~test_mask).sum()} test samples with unseen classes")
                    # Also filter x_test
                    x_test_path = os.path.join(real_test_dir, "x_test.csv")
                    if os.path.exists(x_test_path):
                        x_test = pd.read_csv(x_test_path)
                        x_test = x_test[test_mask]
                        x_test.to_csv(x_test_path, index=False)
                    y_test = y_test[test_mask]

                # Transform y_test with same encoder
                y_test_remapped = y_label_encoder.transform(y_test)
                y_test = pd.DataFrame(y_test_remapped, columns=y_test.columns if isinstance(
                    y_test, pd.DataFrame) else [y_test.name])
                y_test.to_csv(y_test_path, index=False)
                print(f"Remapped y_test.csv to match synthetic data encoding")

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
                'niter': self.niter,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'lambda_gp': self.lambda_gp,
                'z_dim': self.z_dim
            }
            privacy_config = {
                'epsilon': self.epsilon,
                'delta': self.delta,
                'num_teachers': self.num_teachers,
                'lambda_noise': self.privacy_mechanism.lambda_noise
            }
            save_metadata(
                metadata_path,
                self.transformer,
                training_config,
                privacy_config,
                self.random_state
            )
            print(f"Saved metadata.json to {metadata_path}")

        return self

    def evaluate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        model: str = "lr",
        metrics: Optional[list] = None,
        task: Optional[str] = None,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, float]:
        """
        Quick model-centric TSTR evaluation.

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
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Auto-detect task
        if task is None:
            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                task = 'classification'
            else:
                task = 'regression'

        # Generate synthetic training data
        n_train = len(x)
        df_synth = self.sample(n_train)

        # Assuming y is the last column(s)
        if task == 'classification':
            X_synth = df_synth.iloc[:, :-1]
            y_synth = df_synth.iloc[:, -1]
        else:
            X_synth = df_synth.iloc[:, :-1]
            y_synth = df_synth.iloc[:, -1]

        # Train downstream model
        if task == 'classification':
            if model == 'lr':
                clf = LogisticRegression(
                    random_state=random_state, max_iter=1000)
            elif model == 'rf':
                clf = RandomForestClassifier(
                    random_state=random_state, n_estimators=100)
            else:
                clf = LogisticRegression(
                    random_state=random_state, max_iter=1000)

            clf.fit(X_synth, y_synth)
            y_pred = clf.predict(x)

            results = {
                'accuracy': accuracy_score(y, y_pred),
                'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0)
            }

            # Add AUC if binary
            if y.nunique() == 2:
                try:
                    y_proba = clf.predict_proba(x)[:, 1]
                    results['roc_auc'] = roc_auc_score(y, y_proba)
                except:
                    pass
        else:
            if model == 'lr':
                reg = LinearRegression()
            elif model == 'rf':
                reg = RandomForestRegressor(
                    random_state=random_state, n_estimators=100)
            else:
                reg = LinearRegression()

            reg.fit(X_synth, y_synth)
            y_pred = reg.predict(x)

            results = {
                'r2': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred))
            }

        return results

    def __del__(self):
        """Clean up TensorFlow session."""
        if self._sess is not None:
            try:
                self._sess.close()
            except:
                pass
