import chex
import jax.numpy as jnp
from flax import linen as nn


class TGLFNN(nn.Module):
    """A simple MLP with dropout layers, ReLU activation, and outputting a mean and variance."""

    hidden_dimension: int
    n_hidden_layers: int
    dropout: float
    input_means: chex.Array
    input_stds: chex.Array
    output_mean: float
    output_std: float

    @nn.compact
    def __call__(
        self,
        x,
        deterministic: bool = False,
        standardise_inputs: bool = True,
        standardise_outputs: bool = False,
    ):
        if standardise_inputs:
            # Transform to 0 mean and unit variance
            x = (x - self.input_means) / self.input_stds

        x = nn.Dense(self.hidden_dimension)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = nn.relu(x)
        for _ in range(self.n_hidden_layers):
            x = nn.Dense(self.hidden_dimension)(x)
            x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
            x = nn.relu(x)
        mean_and_var = nn.Dense(2)(x)
        mean = mean_and_var[..., 0]
        var = mean_and_var[..., 1]
        var = nn.softplus(var)

        if not standardise_outputs:
            # Transform back from 0 mean and unit variance
            mean = mean * self.output_std + self.output_mean
            var = var * self.output_std**2

        return jnp.stack([mean, var], axis=-1)


class EnsembleTGLFNN(nn.Module):
    """An ensemble of TGLFNN models."""

    input_means: chex.Array
    input_stds: chex.Array
    output_mean: chex.Array
    output_std: chex.Array
    n_models: int = 5
    hidden_dimension: int = 512
    n_hidden_layers: int = 4
    dropout: float = 0.05

    def setup(
        self,
    ):
        self.models = [
            TGLFNN(
                hidden_dimension=self.hidden_dimension,
                n_hidden_layers=self.n_hidden_layers,
                dropout=self.dropout,
                input_means=self.input_means,
                input_stds=self.input_stds,
                output_mean=self.output_mean,
                output_std=self.output_std,
            )
            for i in range(self.n_models)
        ]

    def __call__(self, x, *args, **kwargs):
        # Shape is batch size x 2 x n_models
        outputs = jnp.stack(
            [model(x, *args, **kwargs) for model in self.models], axis=-1
        )
        # Shape is batch_size
        mean = jnp.mean(outputs[:, 0, :], axis=-1)
        aleatoric_uncertainty = jnp.mean(outputs[:, 1, :], axis=-1)
        epistemic_uncertainty = jnp.var(outputs[:, 0, :], axis=-1)
        return jnp.stack([mean, aleatoric_uncertainty + epistemic_uncertainty], axis=-1)

    def get_params_from_pytorch_state_dict(self, pytorch_state_dict: dict):
        params = {}
        for i in range(self.n_models):
            model_dict = {}
            for j in range(self.n_hidden_layers + 2):  # +2 for input and output layers
                # j*3 to skip dropout and activation
                layer_dict = {
                    "kernel": jnp.array(
                        pytorch_state_dict[f"models.{i}.model.{j*3}.weight"]
                    ).T,
                    "bias": jnp.array(
                        pytorch_state_dict[f"models.{i}.model.{j*3}.bias"]
                    ).T,
                }
                model_dict[f"Dense_{j}"] = layer_dict
            params[f"models_{i}"] = model_dict

        return params
