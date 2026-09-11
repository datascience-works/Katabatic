"""
MedGAN Neural Network Components and Utilities.

Based on:
"Generating Multi-label Discrete Patient Records using
Generative Adversarial Networks"
Choi et al. (2017)
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int = 128,
        latent_dim: int = 128,
        bn_decay: float = 0.99,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim

        self.encoder_layer = nn.Linear(
            input_dim,
            latent_dim,
        )

        self.decoder_layer = nn.Linear(
            latent_dim,
            input_dim,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.tanh(
            self.encoder_layer(x)
        )

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        return torch.sigmoid(
            self.decoder_layer(z)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        z = self.encode(x)

        x_recon = self.decode(z)

        return x_recon, z


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bn_decay: float = 0.99,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_dim = latent_dim

        for _ in range(num_layers):

            self.layers.append(
                nn.Linear(
                    in_dim,
                    hidden_dim,
                    bias=False,
                )
            )

            self.batch_norms.append(
                nn.BatchNorm1d(
                    hidden_dim,
                    momentum=1 - bn_decay,
                )
            )

            in_dim = hidden_dim

        self.output_layer = nn.Linear(
            hidden_dim,
            latent_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        x = z

        for layer, batch_norm in zip(
            self.layers,
            self.batch_norms,
        ):

            residual = x

            x = layer(x)
            x = batch_norm(x)
            x = torch.relu(x)

            if residual.shape == x.shape:
                x = x + residual

        x = self.output_layer(x)

        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        first_hidden = 256
        second_hidden = 128

        layers = [
            nn.Linear(
                input_dim * 2,
                first_hidden,
            ),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(
                nn.Dropout(dropout)
            )

        layers.extend(
            [
                nn.Linear(
                    first_hidden,
                    second_hidden,
                ),
                nn.ReLU(),
            ]
        )

        if dropout > 0:
            layers.append(
                nn.Dropout(dropout)
            )

        layers.append(
            nn.Linear(
                second_hidden,
                1,
            )
        )

        self.model = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_average = torch.mean(
            x,
            dim=0,
            keepdim=True,
        )

        batch_average = batch_average.expand(
            x.size(0),
            -1,
        )

        discriminator_input = torch.cat(
            [
                x,
                batch_average,
            ],
            dim=1,
        )

        return torch.sigmoid(
            self.model(
                discriminator_input
            )
        )


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = "rbf",
    bandwidth: float = 1.0,
) -> torch.Tensor:

    def rbf_kernel(
        x,
        y,
        bandwidth,
    ):
        xx = torch.mm(
            x,
            x.t(),
        )

        yy = torch.mm(
            y,
            y.t(),
        )

        xy = torch.mm(
            x,
            y.t(),
        )

        x_sqnorms = torch.diag(xx)
        y_sqnorms = torch.diag(yy)

        k_xx = torch.exp(
            -(
                x_sqnorms.unsqueeze(1)
                + x_sqnorms.unsqueeze(0)
                - 2 * xx
            )
            / (2 * bandwidth**2)
        )

        k_yy = torch.exp(
            -(
                y_sqnorms.unsqueeze(1)
                + y_sqnorms.unsqueeze(0)
                - 2 * yy
            )
            / (2 * bandwidth**2)
        )

        k_xy = torch.exp(
            -(
                x_sqnorms.unsqueeze(1)
                + y_sqnorms.unsqueeze(0)
                - 2 * xy
            )
            / (2 * bandwidth**2)
        )

        return (
            k_xx,
            k_yy,
            k_xy,
        )

    if kernel == "rbf":
        k_xx, k_yy, k_xy = rbf_kernel(
            x,
            y,
            bandwidth,
        )
    else:
        k_xx = torch.mm(
            x,
            x.t(),
        )

        k_yy = torch.mm(
            y,
            y.t(),
        )

        k_xy = torch.mm(
            x,
            y.t(),
        )

    m = x.shape[0]
    n = y.shape[0]

    mmd = (
        k_xx.sum() / (m * m)
        + k_yy.sum() / (n * n)
        - 2 * k_xy.sum() / (m * n)
    )

    return mmd


def sample_noise(
    batch_size: int,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:

    return torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )