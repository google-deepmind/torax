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
              2.82691562998223e-02,
              2.58294982485687e-01,
              7.51532731064005e-01,
              1.52898638812398e00,
              2.61597539330777e00,
              4.10955734212211e00,
              6.08340992472874e00,
              8.54828085960123e00,
              1.14180751339249e01,
              1.45448690085190e01,
              1.78016419395054e01,
              2.11095105087725e01,
              2.44259250193576e01,
              2.77169841485157e01,
              3.09538089854500e01,
              3.41121726512746e01,
              3.71726376565201e01,
              4.01195454787118e01,
              4.29382385995525e01,
              4.56175767225333e01,
              4.81440970724439e01,
              5.04925801508127e01,
              5.26679900645839e01,
              5.47613521745655e01,
              5.67944467049695e01,
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(50.417485084359726),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      0.0,
      5.75064565464661,
      12.33094371445794,
      19.43634142649927,
      27.17472512959477,
      37.33954872035863,
      49.34631456516576,
      61.62177337181222,
      71.7448568580918,
      78.16984686485159,
      81.41932327466112,
      82.69671423167608,
      82.91036276462754,
      82.27647822895267,
      80.92062092335786,
      78.95909164561567,
      76.51162513113849,
      73.67269555479083,
      70.46732802101872,
      66.98345307451862,
      63.16300874776495,
      58.71207695922127,
      54.38524784427834,
      52.33405274954137,
      50.82736326009893,
      50.41748508435973,
  ]).astype('float64')
  jtot = np.array([
      795917.9083302062,
      840782.4499447887,
      907361.0199350917,
      997214.1844258476,
      1147368.4475644985,
      1299436.7025480503,
      1355582.6447244585,
      1270699.3449932635,
      1082130.7974471736,
      883946.832547368,
      736175.4601967924,
      636551.6005556053,
      560567.6935966521,
      494898.79834195157,
      438073.51598169986,
      390326.22609503136,
      349883.16942411405,
      315812.03175793774,
      287118.2725221936,
      257945.50531862015,
      242596.7573229568,
      282222.4593829316,
      394854.20010322006,
      520292.415208095,
      580718.6763027112,
  ])
  s = np.array([
      -0.07213558746176,
      -0.07213558746176,
      -0.10991003958013,
      -0.14562624389212,
      -0.31798201650368,
      -0.48489144551329,
      -0.47719517520208,
      -0.27334155928272,
      0.07788007874047,
      0.44735990737644,
      0.73241268296836,
      0.917711631104,
      1.05655725949415,
      1.19644939698028,
      1.34268091118458,
      1.49699333857542,
      1.64867425366475,
      1.82182056987522,
      2.01325973255752,
      2.23449608843021,
      2.56911309068007,
      2.89960958458338,
      2.6288582770712,
      2.08242930227477,
      1.74940855765015,
      1.81507856025273,
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
              3.60277713715760e-02,
              3.29185366436919e-01,
              9.57794747245436e-01,
              1.94862455169660e00,
              3.33394326962468e00,
              5.23744645188747e00,
              7.75303300895049e00,
              1.08944004257979e01,
              1.45518244713624e01,
              1.85367829768494e01,
              2.26873939580287e01,
              2.69031240377243e01,
              3.11297455362048e01,
              3.53240527386590e01,
              3.94492407689661e01,
              4.34744335569566e01,
              4.73748588943901e01,
              5.11305606969125e01,
              5.47228586150829e01,
              5.81375548408243e01,
              6.13574916711564e01,
              6.43505279867828e01,
              6.71229903192826e01,
              6.97908864070344e01,
              7.23819741685979e01,
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(64.25482269382654),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      0.0,
      7.32893987663357,
      15.71523452021292,
      24.77074511127902,
      34.63296794820204,
      47.58757955656982,
      62.88966392657549,
      78.5341854211854,
      91.4356011391122,
      99.62396263717564,
      103.76527452948281,
      105.39325199238804,
      105.66553746201474,
      104.85768006135298,
      103.12970075767804,
      100.62981969976352,
      97.51063343583776,
      93.89254506305899,
      89.80744795425916,
      85.36740564353558,
      80.49842075830291,
      74.82590789066083,
      69.31155831249427,
      66.69740219379392,
      64.7771940390875,
      64.25482269382653,
  ]).astype('float64')
  jtot = np.array([
      1014361.6642723732,
      1071539.5096540153,
      1156390.909972325,
      1270904.7367364431,
      1462269.608239906,
      1656073.7764759487,
      1727629.2607189012,
      1619449.3036113628,
      1379127.150137565,
      1126550.5786547544,
      938222.5945491319,
      811256.4551907588,
      714418.374888706,
      630726.3142071059,
      558305.0413793066,
      497453.2626631715,
      445910.40146656823,
      402488.2651570188,
      365919.3563938611,
      328739.97346724605,
      309178.6827883966,
      359679.8622050641,
      503223.9623832343,
      663089.3395851713,
      740099.8982473996,
  ])
  s = np.array([
      -0.07213558746176,
      -0.07213558746176,
      -0.10991003958012,
      -0.14562624389212,
      -0.31798201650368,
      -0.48489144551329,
      -0.47719517520207,
      -0.27334155928274,
      0.07788007874055,
      0.4473599073764,
      0.73241268296826,
      0.91771163110416,
      1.05655725949404,
      1.19644939698014,
      1.34268091118474,
      1.49699333857536,
      1.64867425366483,
      1.82182056987507,
      2.01325973255764,
      2.2344960884306,
      2.56911309067965,
      2.89960958458354,
      2.62885827707079,
      2.0824293022747,
      1.74940855765083,
      1.81507856025345,
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
