.. _interfacing_with_surrogates:

JAX-compatible interfaces with ML-surrogates of physics models
##############################################################

This section discusses a variety of options for building JAX-friendly interfaces
to surrogate models.

As an illustrative example, suppose we have a new neural network surrogate
transport model that we would like to use in TORAX. Assume that all the
boilerplate described in the previous sections has been taken care of, as well
as the definition of some functions to convert between TORAX structures and
tensors for the neural network.

.. code-block:: python

    class MyCustomSurrogateTransportModel(TransportModel):
      ...
      def _call_implementation(
          self,
          runtime_params: runtime_params_slice.RuntimeParams,
          geo: geometry.Geometry,
          core_profiles: state.CoreProfiles,
      ) -> TurbulentTransport:
        input_tensor = self._prepare_input(runtime_params, geo, core_profiles)

        output_tensor = self._call_surrogate_model(input_tensor)

        chi_i, chi_e, d_e, v_e = self._parse_output(output_tensor)

        return TurbulentTransport(
            chi_face_ion=chi_i,
            chi_face_electron=chi_e,
            d_e=d_e,
            v_e=v_e,
        )

In this guide, we explore a few options for how you could make the
``_call_surrogate_model`` function for an existing surrogate, while maintaining
the full power of JAX:

1. **Manually reimplementing the model in JAX**.
2. **Converting a Pytorch model to a JAX model**.
3. **Using an ONNX model**.

.. note::
    These conversion methods are necessary in order to make an external model
    compatible with JAX's autodiff and JIT functionality, which is required for
    using TORAX's gradient-driven nonlinear solvers (e.g. Newton-Raphson).
    Interfacing with non-differentiable, non-JITtable models is possible
    (for an example, see the |QuaLiKiz| transport model implementation) if the
    linear solver is used. However, note that if the model is called within the
    step function, JIT will need to be disabled with
    ``JAX_DISABLE_JIT=True``.


Option 1: manually reimplementing the model in JAX
==================================================

If the architecture of the surrogate is sufficiently simple, you might consider
reimplementing the model in JAX. The surrogates in TORAX are mostly implemented
using `Flax Linen`_, and can be found in the |fusion_surrogates|_ repository.
If you're not familiar with Flax, you can check out the `Flax documentation`_
on how to define your own models.

Consider a PyTorch neural network,

.. code-block:: python

    import torch

    class PyTorchMLP(torch.nn.Module):
      def __init__(self, hidden_dim: int, n_hidden: int, output_dim: int, input_dim: int):
        super().__init__()
        self.model = torch.nn.Sequential(
          torch.nn.Linear(input_dim, hidden_dim),
          torch.nn.ReLU(),
          *[torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
          ) for _ in range(n_hidden)],
          torch.nn.Linear(hidden_dim, output_dim)
        )

      def forward(self, x):
        return self.model(x)

    torch_model = PyTorchMLP(hidden_dim, n_hidden, output_dim, input_dim)

This model can be replicated in Flax as follows:

.. code-block:: python

    from flax import linen

    class FlaxMLP(linen.Module):
      hidden_dim: int
      n_hidden: int
      output_dim: int
      input_dim: int

    @linen.compact
    def __call__(self, x):
      x = linen.Dense(self.hidden_dim)(x)
      x = linen.relu(x)
      for _ in range(self.n_hidden):
        x = linen.Dense(self.hidden_dim)(x)
        x = linen.relu(x)
      x = linen.Dense(self.output_dim)(x)
      return x

    flax_model = FlaxMLP(hidden_dim, n_hidden, output_dim, input_dim)

As this is only the model architecture, we need to load the trained weights
separately. This can be a bit fiddly as you have to map from the parameter names
in the weights checkpoint file to the parameter names in the Flax model.

For loading weights from a PyTorch checkpoint, you might do something like:

.. code-block:: python

    import torch

    state_dict = torch.load(PYTORCH_CHECKPOINT_PATH)

    params = {}
    for i in range(n_hidden_layers):
      layer_dict = {
        "kernel": jnp.array(
          state_dict[f"model.{i*2}.weight"]
        ).T,
        "bias": jnp.array(
          pytorch_state_dict[f"model.{i*2}.bias"]
        ).T,
      }
      params[f"Dense_{i}"] = layer_dict

    params = {'params': params}

The model can then be called like any Flax model,

.. code-block:: python

    output_tensor = jax.jit(flax_model.apply)(params, input_tensor)


.. warning::
    You need to be very careful when loading from a PyTorch state dict, as
    Flax and PyTorch may have slightly different representations of the weights
    (for example, one could be the transpose of the other). It's worth
    validating the output of your PyTorch model against your JAX model to make
    sure.


Option 2: converting a PyTorch model to a JAX model
===================================================

.. warning::
    The `torch_xla2`_ package is still evolving, which means there may be
    unexpected breaking changes. Some of the methods described in this section
    may become deprecated with little warning.

If your model is in PyTorch, you could also consider using the `torch_xla2`_
package to do the conversion to JAX automatically.

.. code-block:: python

    import torch
    import torch_xla2 as tx

    trained_model = torch.load(PYTORCH_MODEL_PATH, weights_only=False)  # Use weights_only=False if you want to load the full model
    params, jax_model_from_torch = tx.extract_jax(model)

The model can then be called as a pure JAX function:

.. code-block:: python

    output_tensor = jax.jit(jax_model_from_torch)(params, input_tensor)

To remove the need for performing the conversion every time the model is loaded,
you might want to save a JAX-compatible version of the weights and model to
disk:

.. code-block:: python

    import jax
    import numpy as np

    # jax.export uses StableHLO to serialize the model to a binary format
    exported_model = jax.export(jax.jit(jax_model_from_torch))
    with open("model.hlo", "wb") as f:
      f.write(exported_model.serialize())

    # The weights can be saved as numpy arrays
    np.savez("weights.npz", *params)

The model can then be loaded and run as follows:

.. code-block:: python

    # Load the HLO checkpoint
    with open('model.hlo', 'rb') as f:
      model_as_bytes = f.read()
      model = jax.export.deserialize(model_as_bytes)

    # Load the weights
    weights_as_npz = np.load('weights.npz')
    weights = [jnp.array(v) for v in weights_as_npz.values()]


Option 3: using an ONNX model
=============================

The `Open Neural Network Exchange`_ format (ONNX) is a highly interoperable
format for sharing neural network models. ONNX files include the model
architecture and weights bundled together.

An ONNX model can be loaded and called as follows, making sure to specify the
correct input and output node names for your specific model:

.. code-block:: python

    import onnxruntime as ort
    import numpy as np

    s = ort.InferenceSession(ONNX_MODEL_PATH)
    onnx_output_tensor = s.run(
      # Output node names
      ['output1', 'output2'],
      # Mapping from input node names to input tensors
      # NOTE: input tensors must have correct dtype for your specific model
      {'input': np.asarray(input_tensor, dtype=np.float32)},
    )

However, JAX will not be able to differentiate through the InferenceSession.
To convert the ONNX model to a JAX representation, you can use the
`jaxonnxruntime`_ package:

.. code-block:: python

    import jax.numpy as jnp
    from jaxonnxruntime.backend import Backend as ONNXJaxBackend
    import onnx

    onnx_model = onnx.load_model(ONNX_MODEL_PATH)

    jax_model_from_onnx = ONNXJaxBackend.prepare(onnx_model)
    # NOTE: run() returns a list of output tensors, in order of the output nodes
    output_tensors = jax.jit(jax_model_from_onnx.run)({"input": jnp.asarray(input_tensor, dtype=jnp.float32)})


Option 4: using a JAX callback
==============================
For more information see :ref:`using_jax`.


Best practices
==============

**Caching and lazy loading**: Ideally, the model should be constructed and
weights loaded once only, on the first call to the function. The loaded model
should be cached and reused for subsequent calls.

For example, in the ``_combined`` function of the QLKNN transport model (the
function that actually evaluates this model), we have:

.. code-block:: python

    model = get_model(self._model_path)
    ...
    model_output = model.predict(...)

where

.. code-block:: python

    @functools.lru_cache(maxsize=1)
    def get_model(path: str) -> base_qlknn_model.BaseQLKNNModel:
      """Load the model."""
      ...
      return qlknn_10d.QLKNN10D(path)

By decorating with ``functools.lru_cache(maxsize=1)``, the result of this
function - the loaded model - is stored in the cache and is only re-loaded if
the function is called with a different ``path``.

**JITting model calls**: In general, you should make sure that your forward call
of the model is JITted:

.. code-block:: python

    output_tensor = jax.jit(flax_model.apply)(params, input_tensor)  # Good
    output_tensor = flax_model.apply(params, input_tensor)  # Bad

This is vital to ensure fast performance.

..  _Flax Linen: https://flax-linen.readthedocs.io/en/latest/index.html
..  _Flax documentation: https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html#defining-your-own-models
.. _torch_xla2: https://pytorch.org/xla/master/features/stablehlo.html
.. _Open Neural Network Exchange: https://onnx.ai/
.. _jaxonnxruntime: https://github.com/google/jaxonnxruntime
.. |fusion_surrogates| replace:: ``google-deepmind/fusion_surrogates``
.. _fusion_surrogates: https://github.com/google-deepmind/fusion_surrogates
