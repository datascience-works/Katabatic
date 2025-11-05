from .utils import gradient_penalty, wasserstein_loss, sample
from katabatic.models.base_model import Model
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath("."))


class CWGAN(Model):
    """
    Conditional Wasserstein GAN (CWGAN) Model for synthetic data generation.

    This implementation uses Wasserstein distance with gradient penalty (WGAN-GP)
    and conditional generation based on class labels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.check_dependencies()  # Check dependencies on initialization
        self.generator = None
        self.critic = None
        self.latent_dim = 128
        self.critic_steps = 5
        self.gp_weight = 10.0
        self.batch_size = None
        self.epochs = None
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self._feature_encoder = None
        self.input_dim = None
        self.num_classes = None
        self.x_train = None
        self.y_train = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return a list of required dependencies for this model."""
        return ['tensorflow', 'sklearn', 'numpy', 'pandas']

    def _build_generator(self):
        """
        Build the generator network.

        Returns
        -------
        generator : keras.Model
            The generator model
        """
        # Noise input
        noise_input = layers.Input(shape=(self.latent_dim,))
        # Label input
        label_input = layers.Input(shape=(self.num_classes,))

        # Concatenate noise and label
        gen_input = layers.Concatenate()([noise_input, label_input])

        # Hidden layers
        x = layers.Dense(256, activation='relu')(gen_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Output layer
        output = layers.Dense(self.input_dim, activation='tanh')(x)

        generator = keras.Model(
            [noise_input, label_input], output, name='generator')
        return generator

    def _build_critic(self):
        """
        Build the critic (discriminator) network.

        Returns
        -------
        critic : keras.Model
            The critic model
        """
        # Data input
        data_input = layers.Input(shape=(self.input_dim,))
        # Label input
        label_input = layers.Input(shape=(self.num_classes,))

        # Concatenate data and label
        critic_input = layers.Concatenate()([data_input, label_input])

        # Hidden layers
        x = layers.Dense(512, activation='relu')(critic_input)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output layer (no activation for Wasserstein loss)
        output = layers.Dense(1)(x)

        critic = keras.Model([data_input, label_input], output, name='critic')
        return critic

    def fit(self, x, y, batch_size=32, epochs=100, verbose=1):
        '''
        Fit the CWGAN model to the given data.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features)
            Dataset to fit the model.

        y : array_like of shape (n_samples,)
            Label of the dataset.

        batch_size : int, default=32
            Size of the batch to feed the model at each step.

        epochs : int, default=100
            Number of epochs to use during training.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        Returns
        -------
        self : object
            Fitted model.
        '''
        if verbose is None or not isinstance(verbose, int):
            verbose = 1

        # Prepare data
        self.x_train = x if isinstance(x, np.ndarray) else x.values
        self.y_train = self._label_encoder.fit_transform(y).astype(int)

        # Encode features
        self.input_dim = self.x_train.shape[1]
        self.num_classes = len(np.unique(self.y_train))

        # Normalize features
        x_scaled = self._scaler.fit_transform(self.x_train)

        # One-hot encode labels
        y_onehot = tf.keras.utils.to_categorical(
            self.y_train, self.num_classes)

        self.batch_size = batch_size
        self.epochs = epochs

        # Build models
        self.generator = self._build_generator()
        self.critic = self._build_critic()

        # Optimizers
        generator_optimizer = keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        critic_optimizer = keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

        # Training loop
        dataset_size = x_scaled.shape[0]
        batches_per_epoch = dataset_size // batch_size

        # Create progress bar
        epoch_iterator = tqdm(
            range(epochs), desc="Training CWGAN", disable=not verbose)

        for epoch in epoch_iterator:
            epoch_c_loss = []
            epoch_g_loss = []

            for batch in range(batches_per_epoch):
                # Get random batch
                idx = np.random.randint(0, dataset_size, batch_size)
                real_samples = tf.convert_to_tensor(
                    x_scaled[idx], dtype=tf.float32)
                real_labels = tf.convert_to_tensor(
                    y_onehot[idx], dtype=tf.float32)

                # Train critic
                for _ in range(self.critic_steps):
                    with tf.GradientTape() as tape:
                        # Generate fake samples
                        noise = tf.random.normal([batch_size, self.latent_dim])
                        fake_samples = self.generator(
                            [noise, real_labels], training=True)

                        # Get critic predictions
                        real_output = self.critic(
                            [real_samples, real_labels], training=True)
                        fake_output = self.critic(
                            [fake_samples, real_labels], training=True)

                        # Calculate Wasserstein loss
                        c_loss = tf.reduce_mean(
                            fake_output) - tf.reduce_mean(real_output)

                        # Add gradient penalty
                        gp = gradient_penalty(self.critic, real_samples, fake_samples,
                                              real_labels, batch_size)
                        c_loss += self.gp_weight * gp

                    # Update critic
                    c_gradients = tape.gradient(
                        c_loss, self.critic.trainable_variables)
                    critic_optimizer.apply_gradients(
                        zip(c_gradients, self.critic.trainable_variables))

                epoch_c_loss.append(c_loss.numpy())

                # Train generator
                with tf.GradientTape() as tape:
                    noise = tf.random.normal([batch_size, self.latent_dim])
                    fake_samples = self.generator(
                        [noise, real_labels], training=True)
                    fake_output = self.critic(
                        [fake_samples, real_labels], training=True)

                    # Generator loss (negative of critic's prediction)
                    g_loss = -tf.reduce_mean(fake_output)

                # Update generator
                g_gradients = tape.gradient(
                    g_loss, self.generator.trainable_variables)
                generator_optimizer.apply_gradients(
                    zip(g_gradients, self.generator.trainable_variables))

                epoch_g_loss.append(g_loss.numpy())

            # Update progress bar with loss values
            if verbose:
                avg_c_loss = np.mean(epoch_c_loss)
                avg_g_loss = np.mean(epoch_g_loss)
                epoch_iterator.set_postfix({
                    'C_loss': f'{avg_c_loss:.4f}',
                    'G_loss': f'{avg_g_loss:.4f}'
                })

        self.is_fitted = True
        return self

    def evaluate(self, x, y, model='lr') -> float:
        """
        Perform a TSTR (Training on Synthetic data, Testing on Real data) evaluation.

        Parameters
        ----------
        x, y : array_like
            Test dataset.

        model : str or object
            The model used for evaluation. Should be one of ['lr', 'mlp', 'rf'], 
            or a model class that has sklearn-style `fit` and `predict` methods.

        Returns
        -------
        accuracy_score : float
            The accuracy score of the evaluation.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        eval_model = None
        models = dict(
            lr=LogisticRegression(max_iter=1000),
            rf=RandomForestClassifier(),
            mlp=MLPClassifier(max_iter=1000)
        )
        if model in models.keys():
            eval_model = models[model]()
        elif hasattr(model, 'fit') and hasattr(model, 'predict'):
            eval_model = model
        else:
            raise Exception(
                "Invalid argument `model`. Should be one of ['lr', 'mlp', 'rf'], "
                "or a model class that has sklearn-style `fit` and `predict` methods.")

        # Generate synthetic data
        synthetic_data = self.sample(verbose=0)
        synthetic_x = synthetic_data[:, :-1]
        synthetic_y = synthetic_data[:, -1]

        # Transform test data
        x_test = x if isinstance(x, np.ndarray) else x.values
        x_test_scaled = self._scaler.transform(x_test)
        y_test = self._label_encoder.transform(y)

        # Train on synthetic, test on real
        eval_model.fit(synthetic_x, synthetic_y)
        pred = eval_model.predict(x_test_scaled)

        return accuracy_score(y_test, pred)

    def sample(self, size=None, verbose=1) -> np.ndarray:
        """
        Generate synthetic data.

        Parameters
        ----------
        size : int or None
            Size of the data to be generated. Set to `None` to make the size 
            equal to the size of the training set.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        Returns
        -------
        synthetic_samples : np.ndarray
            Generated synthetic data with features and labels.
        """
        if not self.is_fitted:
            raise Exception("Model must be fitted before sampling.")

        if verbose is None or not isinstance(verbose, int):
            verbose = 1

        sample_size = len(self.x_train) if size is None else size

        # Generate samples for each class proportionally
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        label_proportions = label_counts / len(self.y_train)

        all_samples = []
        all_labels = []

        for label_idx, proportion in zip(unique_labels, label_proportions):
            n_samples = int(sample_size * proportion)
            if n_samples == 0:
                continue

            # Generate noise
            noise = tf.random.normal([n_samples, self.latent_dim])

            # Create label condition
            labels_onehot = tf.keras.utils.to_categorical(
                np.full(n_samples, label_idx), self.num_classes
            )
            # Convert to tensor to match noise tensor type
            labels_onehot = tf.convert_to_tensor(
                labels_onehot, dtype=tf.float32)

            # Generate samples
            fake_samples = self.generator(
                [noise, labels_onehot], training=False)
            fake_samples_np = fake_samples.numpy()

            # Inverse transform to original scale
            fake_samples_original = self._scaler.inverse_transform(
                fake_samples_np)

            all_samples.append(fake_samples_original)
            all_labels.append(np.full(n_samples, label_idx))

        # Combine all samples
        synthetic_x = np.vstack(all_samples)
        synthetic_y = np.concatenate(all_labels)

        # Inverse transform labels
        synthetic_y_original = self._label_encoder.inverse_transform(
            synthetic_y.astype(int))

        # Combine features and labels
        synthetic_data = np.column_stack([synthetic_x, synthetic_y_original])

        if verbose:
            print(f"Generated {len(synthetic_data)} synthetic samples")

        return synthetic_data

    def train(self, dataset, size_category='small', *args, **kwargs):
        """
        Train CWGAN and generate synthetic data.

        Parameters
        ----------
        dataset : str
            Name of the dataset (e.g., 'adult', 'car')
        size_category : str
            Dataset size category ('small', 'medium', 'large')
        *args, **kwargs
            Additional arguments

        Returns
        -------
        self : object
            Trained model
        """
        epochs = 150 if size_category == 'large' else 100

        model_name = "cwgan"
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

        # Set random seed for reproducibility
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        # Load training data
        X = pd.read_csv(x_train_path)
        y = pd.read_csv(y_train_path).values.ravel()
        print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

        self.fit(X, y, epochs=epochs, batch_size=64)

        syn_data = self.sample(X.shape[0])
        df_synth = pd.DataFrame(syn_data)
        x_synth = df_synth.iloc[:, :-1]
        y_synth = df_synth.iloc[:, -1]

        # Set proper column names
        x_synth.columns = X.columns
        y_synth.name = 'target'

        x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"),
                       index=False, header=True)
        print(f"\nSynthetic data saved to: {save_dir}")

        return self
