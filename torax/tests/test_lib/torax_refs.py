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
          torax_mesh=geometry_provider.torax_mesh,
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
      n_rho=25,
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
      dr=geo.drho_norm,
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
      2.66395965857762e06,
      2.63507449221355e06,
      2.57171291638819e06,
      2.49295208016546e06,
      2.39567431739416e06,
      2.29233874073941e06,
      2.24047993512995e06,
      2.36274134758992e06,
      2.69861999396674e06,
      2.96403700886527e06,
      2.75775438144349e06,
      2.11329121928538e06,
      1.45202598088550e06,
      1.02793892445197e06,
      7.88264599831191e05,
      6.20970331530779e05,
      4.79258604417217e05,
      3.55389076090560e05,
      2.50094418726092e05,
      1.64425696483922e05,
      9.86424343124268e04,
      5.20042032681419e04,
      2.26125288440106e04,
      6.44760156917251e03,
      -8.36884651557607e02,
  ])
  s = np.array([
      0.02093808757028,
      0.02093808757028,
      0.04774965851717,
      0.08294316052347,
      0.12685220196587,
      0.17804524483953,
      0.22235364068391,
      0.1962642835125,
      -0.00411599196654,
      -0.28080378993985,
      -0.25196914699796,
      0.19109434469777,
      0.74816705970459,
      1.13739919564292,
      1.33680275524042,
      1.45834363580827,
      1.5705607183252,
      1.68825511479214,
      1.8097364078738,
      1.93047504448354,
      2.0451383537656,
      2.14802893619621,
      2.23381951144242,
      2.29890754347235,
      2.3474205175178,
      2.52466036017095,
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
  geo = geometry.build_standard_geometry(
      geometry.StandardGeometryIntermediates.from_chease(
          geometry_dir=_GEO_DIRECTORY,
          geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
          n_rho=25,
          Ip_from_parameters=False,
          Rmaj=6.2,
          Rmin=2.0,
          B0=5.3,
      )
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
      dr=geo.drho_norm,
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
      839160.0965218098,
      864851.9422342945,
      919092.3553600532,
      1013723.9182892411,
      1166923.0733103524,
      1309995.8119838932,
      1354133.4900918193,
      1259497.8542251373,
      1069042.9548963327,
      874434.6306661696,
      730455.0879010647,
      632869.0119655379,
      557700.6845818826,
      492604.2142102875,
      436320.6841938545,
      389019.0637479049,
      348800.701868444,
      314962.26984272804,
      286434.1270039312,
      257256.8081710547,
      242328.2153636439,
      282697.06942260603,
      394752.49898599176,
      520019.67038794764,
      580635.5973376503,
  ])
  s = np.array([
      -0.07217180720838,
      -0.07217180720838,
      -0.10995641171146,
      -0.14566885115556,
      -0.31797342623655,
      -0.48503249063559,
      -0.47704510597344,
      -0.27308222545984,
      0.07785374277613,
      0.44732310195043,
      0.73248904694036,
      0.91949929868956,
      1.05876200901699,
      1.19789105387465,
      1.34314529971998,
      1.49390400216031,
      1.65055729918186,
      1.82276908903157,
      2.01352540962771,
      2.23716142870353,
      2.57071577441158,
      2.90186610276457,
      2.62732026047895,
      2.08351729391536,
      1.74889986770246,
      1.81817284827307,
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
  geo = geometry.build_standard_geometry(
      geometry.StandardGeometryIntermediates.from_chease(
          geometry_dir=_GEO_DIRECTORY,
          geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
          n_rho=25,
          Ip_from_parameters=True,
          Rmaj=6.2,
          Rmin=2.0,
          B0=5.3,
      )
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
      dr=geo.drho_norm,
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
      1069471.8930054302,
      1102214.9976678425,
      1171341.970629782,
      1291945.6518146081,
      1487191.1014092225,
      1669530.8877037195,
      1725782.3781578145,
      1605173.5061968667,
      1362447.2820096393,
      1114427.704134508,
      930932.2367101787,
      806563.1612568619,
      710764.5005314804,
      627801.9696808034,
      556071.135908784,
      495787.3428992997,
      444530.8451331861,
      401405.28172175714,
      365047.4436259141,
      327862.25985197444,
      308836.43810973485,
      360284.73140663106,
      503094.34887228254,
      662741.7385489461,
      739994.0178338734,
  ])
  s = np.array([
      -0.07217180720838,
      -0.07217180720838,
      -0.10995641171146,
      -0.14566885115556,
      -0.31797342623656,
      -0.48503249063559,
      -0.47704510597342,
      -0.27308222545986,
      0.0778537427762,
      0.44732310195039,
      0.73248904694027,
      0.91949929868973,
      1.05876200901688,
      1.19789105387452,
      1.34314529972014,
      1.49390400216025,
      1.65055729918193,
      1.82276908903142,
      2.01352540962783,
      2.23716142870391,
      2.57071577441116,
      2.90186610276473,
      2.62732026047854,
      2.0835172939153,
      1.74889986770314,
      1.81817284827379,
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
