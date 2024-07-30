# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared setup code for unit tests using reference values."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import numpy as jnp
import numpy as np
import torax
from torax.config import build_sim
from torax.config import config_args
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import runtime_params as sources_params

_GEO_DIRECTORY = 'torax/data/third_party/geo'

# It's best to import the parent `torax` package because that has the
# __init__ file that configures jax to float64
fvm = torax.fvm
geometry = torax.geometry


@chex.dataclass(frozen=True)
class References:
  """Collection of reference values useful for unit tests."""

  runtime_params: general_runtime_params.GeneralRuntimeParams
  geometry_provider: torax.GeometryProvider
  psi: fvm.cell_variable.CellVariable
  psi_face_grad: np.ndarray
  jtot: np.ndarray
  s: np.ndarray


def build_consistent_dynamic_runtime_params_slice_and_geometry(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
    geometry_provider: torax.GeometryProvider,
    sources: dict[str, sources_params.RuntimeParams] | None = None,
    t: chex.Numeric | None = None,
) -> tuple[runtime_params_slice.DynamicRuntimeParamsSlice, geometry.Geometry]:
  """Builds a consistent Geometry and a DynamicRuntimeParamsSlice."""
  t = runtime_params.numerics.t_initial if t is None else t
  return torax.get_consistent_dynamic_runtime_params_slice_and_geometry(
      t,
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params,
          transport_getter=lambda: None,
          sources_getter=lambda: sources,
          stepper_getter=lambda: None,
      ),
      geometry_provider,
  )


def circular_references() -> References:
  """Reference values for circular geometry."""
  # Hard-code the parameters relevant to the tests, so the reference values
  # will stay valid even if we change the Config constructor defaults
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  runtime_params = config_args.recursive_replace(
      runtime_params,
      **{
          'profile_conditions': {
              'Ip': 15,
              'nu': 3,
          },
          'numerics': {
              'q_correction_factor': 1.0,
          },
      },
  )
  geo = geometry.build_circular_geometry(
      nr=25,
      kappa=1.72,
      hires_fac=4,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  )
  # ground truth values copied from example executions using
  # array.astype(str),which allows fully lossless reloading
  psi = fvm.cell_variable.CellVariable(
      value=jnp.array(
          np.array([
              5.20759356768568e-02,
              4.61075402400392e-01,
              1.26170727315354e00,
              2.43206765596828e00,
              3.94565081511133e00,
              5.77184367105234e00,
              7.87766052353025e00,
              1.02382153560177e01,
              1.28750029143908e01,
              1.58977219219003e01,
              1.94142582890593e01,
              2.33373087957383e01,
              2.74144266887867e01,
              3.14433153786909e01,
              3.53392457627434e01,
              3.90737894785106e01,
              4.26319591360467e01,
              4.60025588593374e01,
              4.91777879485926e01,
              5.21537175333016e01,
              5.49305317676010e01,
              5.75124677275163e01,
              5.99074524569073e01,
              6.21264353189880e01,
              6.41824136951859e01,
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(47.64848792277505),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      0.0,
      10.22498666808838,
      20.01579676882877,
      29.25900957036843,
      37.83957897857617,
      45.65482139852536,
      52.64542131194783,
      59.01387081218714,
      65.91968895932578,
      75.56797518773858,
      87.9134091789751,
      98.07626266697432,
      101.9279473262097,
      100.72221724760463,
      97.39825960131307,
      93.36359289418023,
      88.95424143840297,
      84.26499308226755,
      79.38072723137921,
      74.39823961772447,
      69.42035585748502,
      64.54839899788283,
      59.87461823477656,
      55.47457155201538,
      51.39945940494925,
      47.64848792277505,
  ]).astype('float64')
  jtot = np.array([
      2.64483089907557e06,
      2.62207538808562e06,
      2.57142144130046e06,
      2.49759547046357e06,
      2.40264405659704e06,
      2.29858022546275e06,
      2.23801941056702e06,
      2.34053106948193e06,
      2.66369095021464e06,
      2.95110967233083e06,
      2.78563994670315e06,
      2.15635396997821e06,
      1.48164624485340e06,
      1.04259728766598e06,
      7.96665033688655e05,
      6.27464462504372e05,
      4.84740504105495e05,
      3.59945775079680e05,
      2.53744804915958e05,
      1.67210079945156e05,
      1.00629079734462e05,
      5.32908028866413e04,
      2.33275885278276e04,
      6.80081337799893e03,
      -7.57564889674478e02,
  ])
  s = np.array([
      0.02123115567002,
      0.02123115567002,
      0.04807473183537,
      0.08330962515023,
      0.1272616961616,
      0.17849347063073,
      0.22280866177671,
      0.19660028856322,
      -0.00412201341123,
      -0.28116059272249,
      -0.25225085163436,
      0.19128432029727,
      0.74883427170789,
      1.13831587807874,
      1.33778265626641,
      1.45932129634624,
      1.57152834198791,
      1.68921513855111,
      1.81068990440061,
      1.93142068620783,
      2.04607268093229,
      2.14894676653499,
      2.23471455086914,
      2.29977334955224,
      2.34825332990576,
      2.52550582782645,
  ])
  return References(
      runtime_params=runtime_params,
      geometry_provider=torax.ConstantGeometryProvider(geo),
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
  )


def chease_references_Ip_from_chease() -> References:  # pylint: disable=invalid-name
  """Reference values for CHEASE geometry where the Ip comes from the file."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  runtime_params = config_args.recursive_replace(
      runtime_params,
      **{
          'profile_conditions': {
              'Ip': 15,
              'nu': 3,
          },
          'numerics': {
              'q_correction_factor': 1.0,
          },
      },
  )
  geo = build_sim.build_chease_geometry(
      geometry_dir=_GEO_DIRECTORY,
      geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      nr=25,
      Ip_from_parameters=False,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  )
  # ground truth values copied from an example PINT execution using
  # array.astype(str),which allows fully lossless reloading
  psi = fvm.cell_variable.CellVariable(
      value=jnp.array(
          np.array([
              2.82687863338609e-02,
              2.58298236304414e-01,
              7.51579093782739e-01,
              1.52901201995250e00,
              2.61601628035973e00,
              4.10972202022658e00,
              6.08320997251803e00,
              8.54844317927733e00,
              1.14179461025201e01,
              1.45457922684783e01,
              1.78021470093261e01,
              2.11100490363611e01,
              2.44258247198332e01,
              2.77171992814264e01,
              3.09539363677558e01,
              3.41125175970711e01,
              3.71729003017132e01,
              4.01203899874561e01,
              4.29387769844556e01,
              4.56178731410229e01,
              4.81440898985093e01,
              5.04919539282557e01,
              5.26675847767664e01,
              5.47608052188134e01,
              5.67947370368951e01,
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(50.417485084359726),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      0.0,
      5.75073624926384,
      12.33202143695811,
      19.43582315424405,
      27.1751065101808,
      37.34264349667116,
      49.33719880728622,
      61.63083016898254,
      71.73757308106956,
      78.19615414895576,
      81.40886852119453,
      82.69755067587425,
      82.89439208680163,
      82.28436403982995,
      80.91842715823611,
      78.96453073288274,
      76.50956761605237,
      73.68724214357165,
      70.45967492498733,
      66.97740391418456,
      63.15541893715846,
      58.69660074366081,
      54.3907712127659,
      52.33051105117621,
      50.84829545204244,
      50.41748508435973,
  ]).astype('float64')
  jtot = np.array([
      795943.2757002946,
      840809.6548931613,
      907355.2753060618,
      997252.7860353531,
      1147287.5557633871,
      1299375.0237769606,
      1355749.626520751,
      1271033.7368197236,
      1082400.964845261,
      883541.529544109,
      735645.0288139464,
      636538.4992467065,
      560944.0368406302,
      495150.44043368497,
      438002.1010280469,
      390492.95132961264,
      349968.695608343,
      315226.96354638913,
      286654.6355284753,
      257677.72282051452,
      242664.07572776848,
      282847.79237364535,
      395745.2569511145,
      521042.01033411804,
      579200.1679234514,
  ])
  s = np.array([
      -0.07221240050235,
      -0.07221240050235,
      -0.10974521889679,
      -0.14566570673946,
      -0.31821045423392,
      -0.48437494158659,
      -0.47786881885484,
      -0.27296558083396,
      0.07676689980166,
      0.44758291747056,
      0.73338472251048,
      0.91916612522922,
      1.05841439699259,
      1.19427091689405,
      1.34225560062758,
      1.49556452173006,
      1.64916460364498,
      1.8221519187211,
      2.01686015385676,
      2.236358704094,
      2.57055418460317,
      2.89810523377755,
      2.62431676142289,
      2.0786251004254,
      1.74669124535595,
      1.81634615194803,
  ])
  return References(
      runtime_params=runtime_params,
      geometry_provider=torax.ConstantGeometryProvider(geo),
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
  )


def chease_references_Ip_from_runtime_params() -> References:  # pylint: disable=invalid-name
  """Reference values for CHEASE geometry where the Ip comes from the config."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  runtime_params = config_args.recursive_replace(
      runtime_params,
      **{
          'profile_conditions': {
              'Ip': 15,
              'nu': 3,
          },
          'numerics': {
              'q_correction_factor': 1.0,
          },
      },
  )
  geo = build_sim.build_chease_geometry(
      geometry_dir=_GEO_DIRECTORY,
      geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      nr=25,
      Ip_from_parameters=True,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  )
  # ground truth values copied from an example executions using
  # array.astype(str),which allows fully lossless reloading
  psi = fvm.cell_variable.CellVariable(
      value=jnp.array(
          np.array([
              3.60272998665571e-02,
              3.29189513282900e-01,
              9.57853834450339e-01,
              1.94865721831205e00,
              3.33399537833796e00,
              5.23765632674312e00,
              7.75277817882905e00,
              1.08946072949424e01,
              1.45516600266262e01,
              1.85379596302444e01,
              2.26880376468546e01,
              2.69038103669742e01,
              3.11296177089611e01,
              3.53243269159014e01,
              3.94494031119997e01,
              4.34748731748978e01,
              4.73751936239048e01,
              5.11316369854171e01,
              5.47235447624864e01,
              5.81379326128902e01,
              6.13574825282986e01,
              6.43497298942302e01,
              6.71224737981430e01,
              6.97901893364682e01,
              7.23823441836110e01,
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(64.25482269382654),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      0.0,
      7.32905533540858,
      15.71660802918596,
      24.77008459654269,
      34.6334540006478,
      47.59152371012898,
      62.87804630214837,
      78.54572790283396,
      91.42631829209348,
      99.65749009045574,
      103.75195041525585,
      105.39431800299042,
      105.64518354967092,
      104.8677301735089,
      103.12690490245639,
      100.63675157245378,
      97.50801122517369,
      93.91108403780848,
      89.79769442673326,
      85.35969626009319,
      80.48874788521178,
      74.80618414828939,
      69.31859759781887,
      66.69288845813028,
      64.8038711785695,
      64.25482269382653,
  ]).astype('float64')
  jtot = np.array([
      1014393.9938474118,
      1071574.1811401332,
      1156383.5887003709,
      1270953.9327558433,
      1462166.5152686965,
      1655995.1696494902,
      1727842.0715264492,
      1619875.4709913302,
      1379471.4663649946,
      1126034.038161414,
      937546.5835502037,
      811239.7581604314,
      714898.0074679875,
      631047.0207628217,
      558214.0262250169,
      497665.74649424717,
      446019.4007510254,
      401742.6219713411,
      365328.471847233,
      328398.6966876795,
      309264.4770749796,
      360476.8210453723,
      504359.5743573825,
      664044.6649418375,
      738164.6274478296,
  ])
  s = np.array([
      -0.07221240050235,
      -0.07221240050235,
      -0.1097452188968,
      -0.14566570673946,
      -0.31821045423391,
      -0.48437494158659,
      -0.47786881885484,
      -0.27296558083405,
      0.07676689980166,
      0.44758291747071,
      0.73338472251056,
      0.91916612522911,
      1.05841439699259,
      1.19427091689393,
      1.34225560062751,
      1.49556452173017,
      1.64916460364509,
      1.82215191872132,
      2.01686015385609,
      2.23635870409426,
      2.57055418460362,
      2.89810523377699,
      2.62431676142303,
      2.07862510042521,
      1.74669124535614,
      1.81634615194823,
  ])
  return References(
      runtime_params=runtime_params,
      geometry_provider=torax.ConstantGeometryProvider(geo),
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
  )


class ReferenceValueTest(parameterized.TestCase):
  """Unit using reference values from previous executions."""

  def setUp(self):
    super().setUp()
    # Some pre-calculated reference values are used in more than one test.
    # These are loaded here.
    self.circular_references = circular_references()
    # pylint: disable=invalid-name
    self.chease_references_with_Ip_from_chease = (
        chease_references_Ip_from_chease()
    )
    self.chease_references_with_Ip_from_config = (
        chease_references_Ip_from_runtime_params()
    )
    # pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
