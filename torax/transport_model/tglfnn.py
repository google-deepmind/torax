import chex
import jax.numpy as jnp
from flax import nnx


@chex.dataclass(frozen=True)
class TGLFInputs:
    r"""Dimensionless inputs to the TGLF model.

    Attributes:
    -----------
      Te_grad: chex.Array
        Normalized electron temperature gradient: :math:`-{\frac {a}{T_{e}}}{\frac {dT_{e}}{dr}}`

      Ti_grad: chex.Array
        Normalized ion temperature gradient: :math:`-{\frac {a}{T_{i}}}{\frac {dT_{i}}{dr}}`

      Ti_over_Te: chex.Array
        Temperature ratio: :math:`{\frac {T_{i}}{T_{e}}}`

      rmin: chex.Array
        Flux surface centroid minor radius: :math:`\frac{r}{a}`

      dRmaj: chex.Array
        :math:`{\frac {\partial R_{maj}}{\partial x}}`

      q: chex.Array
        Safety factor, :math:`q`

      s_hat: chex.Array
        s_hat = r/q * dq/dr

      nu_ee: chex.Array
        Electron-electron collision frequency

      kappa: chex.Array
        Elongation of flux surface

      kappa_shear: chex.Array
        Shear in elongation: :math:`{\frac {r}{\kappa }}{\frac {\partial \kappa }{\partial r}}`

      delta: chex.Array
        Triangularity of flux surface

      delta_shear: chex.Array
        Shear in triangularity of flux surface: :math:`r{\frac {\partial \delta }{\partial r}}`

      beta_e: chex.Array
        :math:`\beta_e:math:` defined w.r.t :math:`B_\mathrm{unit}`

      Zeff: chex.Array
        Effective ion charge
    """

    Te_grad_norm: chex.Array
    Ti_grad_norm: chex.Array
    Ti_over_Te: chex.Array
    Rmin: chex.Array
    dRmaj: chex.Array
    q: chex.Array
    s_hat: chex.Array
    ei_collision_freq: chex.Array
    kappa: chex.Array
    kappa_shear: chex.Array
    delta: chex.Array
    delta_shear: chex.Array
    beta_e: chex.Array
    Zeff: chex.Array


class TGLFNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        input_dimension = 15
        output_dimension = 2
        n_hidden_layers = 6
        hidden_dimension = 512
        dropout = 0.05
        self.input_layer = nnx.Linear(input_dimension, hidden_dimension, rngs=rngs)
        self.input_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.hidden_layers = [
            nnx.Linear(hidden_dimension, hidden_dimension, rngs=rngs)
            for _ in range(n_hidden_layers)
        ]
        self.dropout_layers = [
            nnx.Dropout(dropout, rngs=rngs) for _ in range(n_hidden_layers)
        ]
        self.output_layer = nnx.Linear(hidden_dimension, output_dimension, rngs=rngs)

    def __call__(self, x):
        x = self.input_layer(x)
        x = self.input_dropout(x)
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x)
        mean, var = self.output_layer(x)
        var = nnx.softplus(var)
        return jnp.hstack([mean, var])


class EnsembleTGLFNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, n_models: int = 5):
        self.models = [TGLFNN(rngs) for _ in range(n_models)]
        self.n_models = n_models

    @nnx.jit
    def __call__(self, x):
        outputs = jnp.stack([model(x) for model in self.models])
        mean = jnp.mean(outputs[..., 0], axis=-1)
        aleatoric_uncertainty = jnp.mean(outputs[..., 1], axis=-1)
        epistemic_uncertainty = jnp.var(outputs[..., 0], axis=-1)
        return jnp.hstack([mean, aleatoric_uncertainty + epistemic_uncertainty])
