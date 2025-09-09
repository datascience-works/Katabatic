"""
PATE-GAN implementation by same last author as the original paper from 2023.
source: https://github.com/vanderschaarlab/synthcity/blob/41e6e5acfd886dd4ebc0528039e9395a2a93b380/src/synthcity/plugins/privacy/plugin_pategan.py
"""

# """
# Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees," International Conference on Learning Representations (ICLR), 2019.
# Paper link: https://openreview.net/forum?id=S1zk9iRqF7
# """


# stdlib
# from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from collections import defaultdict

# Necessary packages
import torch
from torch.utils.data import DataLoader

# Necessary packages
from pydantic import validate_arguments
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# synthcity absolute
import synthcity.logger as log
# from synthcity.plugins.core.dataloader import DataLoader
# from synthcity.plugins.core.distribution import (
#     CategoricalDistribution,
#     Distribution,
#     FloatDistribution,
#     IntegerDistribution,
# )
# from synthcity.plugins.core.plugin import Plugin
# from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.serializable import Serializable
from synthcity.utils.constants import DEVICE

from pate_gans.synthcity.tabular_gan import TabularGAN


class Teachers(Serializable):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        num_teachers: int,
        samples_per_teacher: int,
        batch_size: int,
        lamda: float = 1e-3,  # PATE noise size
        template: str = "xgboost",
    ) -> None:
        super().__init__()

        self.num_teachers = num_teachers
        self.teachers_seen_data = defaultdict(set)
        self.samples_per_teacher = samples_per_teacher
        self.batch_size = batch_size
        self.lamda = lamda
        self.model_args: dict = {}
        if template == "xgboost":
            self.model_template = XGBClassifier
            self.model_args = {
                "verbosity": 0,
                "depth": 3,
                "nthread": 2,
            }
        else:
            self.model_template = LogisticRegression
            self.model_args = {
                "solver": "sag",
                "max_iter": 10000,
            }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: np.ndarray, generator: Any, add_X_index, return_teachers_loaders=False) -> Any:
        # 1. train teacher models
        self.teacher_models: list = []

        permutations = np.random.permutation(len(X))

        if return_teachers_loaders:
            teachers_loaders = {}

        for tidx in range(self.num_teachers):
            teacher_idx = permutations[int(tidx * self.samples_per_teacher): int((tidx + 1) * self.samples_per_teacher)]
            teacher_X = X[teacher_idx, :]

            # NOTE: reduce size to batch size (to be consistent with PATE-GAN paper, L12, Alg1)
            g_mb = np.asarray(generator(self.batch_size))

            # BUG!!!
            idx = np.random.permutation(len(teacher_X[:, 0]))
            x_mb = teacher_X[idx[: self.samples_per_teacher], :]
            if return_teachers_loaders:
                teachers_loaders[tidx] = x_mb
            # NOTE: reduce size to batch size (to be consistent with PATE-GAN paper, L13, Alg1)
            x_mb = x_mb[: self.batch_size, :]
            if add_X_index:
                teach_idx, x_mb = x_mb[:, 0], x_mb[:, 1:]
                teach_idx = set(teach_idx.astype(int))
                self.teachers_seen_data[tidx].update(teach_idx)

            x_comb = np.concatenate((x_mb, g_mb), axis=0)
            # NOTE: reduce size to batch size (to be consistent with PATE-GAN paper, L12, Alg1)
            y_comb = np.concatenate((np.ones([len(x_mb)]), np.zeros([len(g_mb)])), axis=0)
            model = self.model_template(**self.model_args)
            model.fit(x_comb, y_comb)

            self.teacher_models.append(model)

        if return_teachers_loaders:
            return teachers_loaders
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pate_lamda(self, x: np.ndarray) -> Tuple[int, int, int]:
        """Returns PATE_lambda(x).

        Args:
          - x: feature vector

        Returns:
          - n0, n1: the number of label 0 and 1, respectively
          - out: label after adding laplace noise.
        """

        predictions = []

        for tidx, teacher in enumerate(self.teacher_models):
            y_pred = teacher.predict(x)
            predictions.append(y_pred)

        y_hat = np.vstack(predictions)  # (num_teachers, batch_size)

        n0 = np.sum(y_hat == 0, axis=0)  # (batch_size, )
        n1 = np.sum(y_hat == 1, axis=0)  # (batch_size, )

        lap_noise = np.random.laplace(loc=0.0, scale=1 / self.lamda, size=n0.shape)

        out = (n1 + lap_noise) / (n0 + n1 + 1e-8)  # (batch_size, )
        out = (out > 0.5).astype(int)

        return n0, n1, out


class PG_SYNTHCITY_AUDIT(Serializable):
    """Basic PATE-GAN framework."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # GAN
        max_iter: int = 1000,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 100,
        generator_nonlin: str = "relu",
        generator_n_iter: int = 10,
        generator_dropout: float = 0,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 100,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 200,
        random_state: int = 0,
        clipping_value: int = 1,
        encoder_max_clusters: int = 5,
        device: Any = DEVICE,
        # Privacy
        num_teachers: int = 10,
        teacher_template: str = "linear",
        epsilon: float = 1.0,
        delta: Optional[float] = None,
        lamda: float = 1e-3,
        alpha: int = 100,
        encoder: Any = None,
        record_teachers=False,
    ) -> None:
        super().__init__()

        self.max_iter = max_iter
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.generator_n_iter = generator_n_iter
        self.generator_dropout = generator_dropout
        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.device = device
        # Privacy
        self.num_teachers = num_teachers
        self.teachers_seen_data = defaultdict(set)
        self.teacher_template = teacher_template
        self.epsilon = epsilon
        self.delta = None
        self.lamda = lamda
        self.alpha = alpha
        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.record_teachers = record_teachers
        if self.record_teachers:
            self.teachers_dict = {i: np.zeros([self.max_iter, self.num_teachers]) for i in range(self.num_teachers)}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X_train: pd.DataFrame, add_X_index=False) -> Any:
        self.columns = X_train.columns

        if self.delta is None:
            self.delta = 1 / (len(X_train) * np.sqrt(len(X_train)))

        log.info(f"[pategan] using delta = {self.delta}")
        self.samples_per_teacher = int(len(X_train) / self.num_teachers)
        self.batch_size = min(self.batch_size, self.samples_per_teacher)

        self.model = TabularGAN(
            X_train,
            n_units_latent=self.generator_n_units_hidden,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="none",
            generator_lr=self.lr,
            generator_residual=True,
            generator_n_iter=self.generator_n_iter,
            generator_batch_norm=False,
            generator_dropout=0,
            generator_weight_decay=self.weight_decay,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_batch_norm=False,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            clipping_value=self.clipping_value,
            encoder_max_clusters=self.encoder_max_clusters,
            encoder=self.encoder,
            n_iter_print=self.generator_n_iter - 1,
            device=self.device,
        )
        X_train_enc = self.model.encode(X_train)
        if add_X_index:
            X_train_enc = X_train_enc.reset_index()

        # alpha initialize
        self.alpha_dict = np.zeros([self.alpha])

        # initialize epsilon_hat
        epsilon_hat = 0

        # Iterations
        it = 0
        while epsilon_hat < self.epsilon and it < self.max_iter:
            it += 1

            log.debug(
                f"[pategan it {it}] 1. Train teacher models epsilon_hat = {epsilon_hat}. num_teachers = {self.num_teachers} samples_per_teacher = {self.samples_per_teacher}"
            )

            # 1. Train teacher models
            self.teachers = Teachers(
                num_teachers=self.num_teachers,
                samples_per_teacher=self.samples_per_teacher,
                batch_size=self.batch_size,
                lamda=self.lamda,
                template=self.teacher_template,
            )
            if self.record_teachers and it == 1:
                teachers_loaders = self.teachers.fit(np.asarray(X_train_enc), self.model, add_X_index, return_teachers_loaders=True)
                teachers_loaders = {i: DataLoader(teacher_data, batch_size=len(teacher_data), shuffle=False) for i, teacher_data in teachers_loaders.items()}
            else:
                self.teachers.fit(np.asarray(X_train_enc), self.model, add_X_index)
            if add_X_index:
                for teach_idx, teach_seen_data in self.teachers.teachers_seen_data.items():
                    self.teachers_seen_data[teach_idx].update(teach_seen_data)

            log.debug(f"[pategan it {it}] 2. GAN training")

            # 2. Student training
            def fake_labels_generator(X: torch.Tensor) -> torch.Tensor:
                if self.num_teachers == 0:
                    return torch.zeros((len(X),))

                X_batch = pd.DataFrame(X.detach().cpu().numpy())

                n0_mb, n1_mb, Y_mb = self.teachers.pate_lamda(np.asarray(X_batch))
                if np.sum(Y_mb) >= len(X) / 2:
                    return torch.zeros((len(X),))

                # Compute alpha
                self._update_alpha(n0_mb, n1_mb)

                # PATE labels for X
                return torch.from_numpy(np.reshape(np.asarray(Y_mb, dtype=int), [-1, 1]))

            _X_train_enc = X_train_enc if not add_X_index else X_train_enc.iloc[:, 1:]
            self.model.fit(_X_train_enc, fake_labels_generator=fake_labels_generator, encoded=True)

            # epsilon_hat computation
            curr_list: List = []
            for lidx in range(self.alpha):
                local_alpha = (self.alpha_dict[lidx] - np.log(self.delta)) / float(lidx + 1)
                curr_list.append(local_alpha)

            epsilon_hat = np.min(curr_list)
            log.info(f"[pategan it {it}] epsilon_hat = {epsilon_hat}. self.epsilon = {self.epsilon}")

            if self.record_teachers:
                if it == 1:
                    teachers_noise = torch.rand(self.samples_per_teacher,
                                                self.model.model.n_units_latent + self.model.model.n_units_conditional,
                                                device=self.device)
                    criterion = torch.nn.BCELoss()

                with torch.no_grad():
                    fake_data = self.model.model.generator(teachers_noise)
                    label_fake = torch.full((len(teachers_noise),), 0, dtype=torch.float, device=self.device)

                    for teacher_id in range(self.num_teachers):
                        teacher_model = self.teachers.teacher_models[teacher_id]

                        for teacher_j in range(self.num_teachers):
                            teacher_data = next(iter(teachers_loaders[teacher_j])).to(self.device)
                            label_real = torch.full((teacher_data.shape[0],), 1, dtype=torch.float, device=self.device)

                            data_combined = torch.cat((teacher_data, fake_data), axis=0)
                            label_combined = torch.cat((label_real, label_fake), axis=0)

                            output_combined = teacher_model.predict_proba(data_combined)[:, 1]
                            loss_combined = criterion(torch.from_numpy(output_combined), label_combined.double())

                            self.teachers_dict[teacher_id][it - 1, teacher_j] = loss_combined
                    print(it, self.teachers_dict[0][it - 1])

        log.debug("pategan training done")
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _update_moments_accountant(self, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
        # Update moments accountant
        qbase = self.lamda * np.abs(n0 - n1)
        return (2 + qbase) / (4 * np.exp(qbase))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _update_alpha(self, n0: np.ndarray, n1: np.ndarray) -> Dict:
        # Update moments accountant
        q = self._update_moments_accountant(n0, n1)
        # Compute alpha
        for lidx in range(self.alpha):
            upper = 2 * self.lamda**2 * (lidx + 1) * (lidx + 2)
            t = (1 - q) * np.power((1 - q) / (1 - np.exp(2 * self.lamda) * q), lidx + 1)
            # BUG!!!
            t = np.log(t + q * np.exp(2 * self.lamda * lidx + 1))
            self.alpha_dict[lidx] += np.clip(t, a_min=0, a_max=upper).sum()
        return self.alpha_dict

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(self, count: int) -> np.ndarray:
        samples = self.model(count)
        return self.model.decode(pd.DataFrame(samples))

    def sd_predict(self, X):
        with torch.no_grad():
            X = self.model.encode(X)
            # cond = ConditionalDatasetSampler(X, self.model.encoder.layout()).get_dataset_conditionals()
            cond = self.model.dataloader_sampler.sample_conditional(len(X), p=self.model.sample_prob)
            X = self.model.model._append_optional_cond(torch.from_numpy(np.array(X)), torch.from_numpy(cond))
            s_predict = torch.sigmoid(self.model.model.discriminator(X))
        return s_predict.detach().numpy()

    def td_predict(self, X):
        with torch.no_grad():
            X = self.model.encode(X)
            t_predict = np.array([teacher.predict_proba(X)[:, 1] for teacher in self.teachers.teacher_models]).mean(axis=0)
        return t_predict

# class PATEGANPlugin(Plugin):
#     """
#     .. inheritance-diagram:: synthcity.plugins.privacy.plugin_pategan.PATEGANPlugin
#         :parts: 1
#
#     PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees.
#
#     Args:
#         generator_n_layers_hidden: int
#             Number of hidden layers in the generator
#         generator_n_units_hidden: int
#             Number of hidden units in each layer of the Generator
#         generator_nonlin: string, default 'leaky_relu'
#             Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
#         n_iter: int
#             Maximum number of iterations in the Generator.
#         generator_dropout: float
#             Dropout value. If 0, the dropout is not used.
#         discriminator_n_layers_hidden: int
#             Number of hidden layers in the discriminator
#         discriminator_n_units_hidden: int
#             Number of hidden units in each layer of the discriminator
#         discriminator_nonlin: string, default 'leaky_relu'
#             Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
#         discriminator_n_iter: int
#             Maximum number of iterations in the discriminator.
#         discriminator_dropout: float
#             Dropout value for the discriminator. If 0, the dropout is not used.
#         lr: float
#             learning rate for optimizer.
#         weight_decay: float
#             l2 (ridge) penalty for the weights.
#         batch_size: int
#             Batch size
#         random_state: int
#             random_state used
#         clipping_value: int, default 0
#             Gradients clipping value. Zero disables the feature
#         teacher_template: str
#             Model to use for the teachers. Can be linear, xgboost.
#         epsilon: float
#             Differential privacy parameter
#         delta: float
#             Differential privacy parameter
#         lambda: float
#             Noise size
#         encoder_max_clusters: int
#             The max number of clusters to create for continuous columns when encoding
#         # Core Plugin arguments
#         workspace: Path.
#             Optional Path for caching intermediary results.
#         compress_dataset: bool. Default = False.
#             Drop redundant features before training the generator.
#         sampling_patience: int.
#             Max inference iterations to wait for the generated data to match the training schema.
#
#     Example:
#         >>> from sklearn.datasets import load_iris
#         >>> from synthcity.plugins import Plugins
#         >>>
#         >>> X, y = load_iris(as_frame = True, return_X_y = True)
#         >>> X["target"] = y
#         >>>
#         >>> plugin = Plugins().get("pategan", n_iter = 100)
#         >>> plugin.fit(X)
#         >>>
#         >>> plugin.generate(50)
#
#
#     """
#
#     @validate_arguments(config=dict(arbitrary_types_allowed=True))
#     def __init__(
#         self,
#         # GAN
#         n_iter: int = 200,
#         generator_n_iter: int = 10,
#         generator_n_layers_hidden: int = 2,
#         generator_n_units_hidden: int = 500,
#         generator_nonlin: str = "relu",
#         generator_dropout: float = 0,
#         discriminator_n_layers_hidden: int = 2,
#         discriminator_n_units_hidden: int = 500,
#         discriminator_nonlin: str = "leaky_relu",
#         discriminator_n_iter: int = 1,
#         discriminator_dropout: float = 0.1,
#         lr: float = 1e-3,
#         weight_decay: float = 1e-3,
#         batch_size: int = 200,
#         random_state: int = 0,
#         clipping_value: int = 1,
#         encoder_max_clusters: int = 5,
#         # Privacy
#         num_teachers: int = 10,
#         teacher_template: str = "xgboost",
#         epsilon: float = 1.0,
#         delta: Optional[float] = None,
#         lamda: float = 1e-3,
#         alpha: int = 100,
#         encoder: Any = None,
#         # core plugin arguments
#         device: Any = DEVICE,
#         workspace: Path = Path("workspace"),
#         compress_dataset: bool = False,
#         sampling_patience: int = 500,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(
#             device=device,
#             random_state=random_state,
#             sampling_patience=sampling_patience,
#             workspace=workspace,
#             compress_dataset=compress_dataset,
#             **kwargs,
#         )
#
#         self.model = PG_SYNTHCITY(
#             max_iter=n_iter,
#             generator_n_layers_hidden=generator_n_layers_hidden,
#             generator_n_units_hidden=generator_n_units_hidden,
#             generator_nonlin=generator_nonlin,
#             generator_n_iter=generator_n_iter,
#             generator_dropout=generator_dropout,
#             discriminator_n_layers_hidden=discriminator_n_layers_hidden,
#             discriminator_n_units_hidden=discriminator_n_units_hidden,
#             discriminator_nonlin=discriminator_nonlin,
#             discriminator_n_iter=discriminator_n_iter,
#             discriminator_dropout=discriminator_dropout,
#             lr=lr,
#             weight_decay=weight_decay,
#             batch_size=batch_size,
#             random_state=random_state,
#             clipping_value=clipping_value,
#             encoder_max_clusters=encoder_max_clusters,
#             encoder=encoder,
#             device=device,
#             # Privacy
#             num_teachers=num_teachers,
#             teacher_template=teacher_template,
#             epsilon=epsilon,
#             delta=delta,
#             lamda=lamda,
#         )
#
#     @staticmethod
#     def name() -> str:
#         return "pategan"
#
#     @staticmethod
#     def type() -> str:
#         return "privacy"
#
#     @staticmethod
#     def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
#         return [
#             IntegerDistribution(name="n_iter", low=1, high=500),
#             IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
#             IntegerDistribution(
#                 name="generator_n_units_hidden", low=50, high=500, step=50
#             ),
#             CategoricalDistribution(
#                 name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
#             ),
#             IntegerDistribution(name="generator_n_iter", low=1, high=100),
#             FloatDistribution(name="generator_dropout", low=0, high=0.2),
#             IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
#             IntegerDistribution(
#                 name="discriminator_n_units_hidden", low=50, high=550, step=50
#             ),
#             CategoricalDistribution(
#                 name="discriminator_nonlin",
#                 choices=["relu", "leaky_relu", "tanh", "elu"],
#             ),
#             IntegerDistribution(name="discriminator_n_iter", low=1, high=5),
#             FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
#             CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
#             CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
#             CategoricalDistribution(name="batch_size", choices=[64, 128, 256, 512]),
#             IntegerDistribution(name="num_teachers", low=5, high=200),
#             CategoricalDistribution(
#                 name="teacher_template", choices=["linear", "xgboost"]
#             ),
#             IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
#             FloatDistribution(name="lamda", low=0, high=1),
#             CategoricalDistribution(
#                 name="delta", choices=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
#             ),
#             IntegerDistribution(name="alpha", low=10, high=500),
#         ]
#
#     def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "PATEGANPlugin":
#         self.model.fit(X.dataframe())
#
#         return self
#
#     def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
#         return self._safe_generate(self.model.sample, count, syn_schema)
#
#
# plugin = PATEGANPlugin
