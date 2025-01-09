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
from torax import fvm
from torax import sim as sim_lib
from torax.config import config_args
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.sources import runtime_params as sources_params
from torax.stepper import runtime_params as stepper_params
from torax.transport_model import runtime_params as transport_model_params

# Internal import.


_GEO_DIRECTORY = 'torax/data/third_party/geo'


@chex.dataclass(frozen=True)
class References:
  """Collection of reference values useful for unit tests."""

  runtime_params: general_runtime_params.GeneralRuntimeParams
  geometry_provider: geometry_provider_lib.GeometryProvider
  psi: fvm.cell_variable.CellVariable
  psi_face_grad: np.ndarray
  jtot: np.ndarray
  s: np.ndarray
  Ip_from_parameters: bool  # pylint: disable=invalid-name


def build_consistent_dynamic_runtime_params_slice_and_geometry(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    sources: dict[str, sources_params.RuntimeParams] | None = None,
    t: chex.Numeric | None = None,
) -> tuple[runtime_params_slice.DynamicRuntimeParamsSlice, geometry.Geometry]:
  """Builds a consistent Geometry and a DynamicRuntimeParamsSlice."""
  t = runtime_params.numerics.t_initial if t is None else t
  return sim_lib.get_consistent_dynamic_runtime_params_slice_and_geometry(
      t,
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params,
          transport=transport_model_params.RuntimeParams(),
          sources=sources,
          stepper=stepper_params.RuntimeParams(),
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
              'Ip_tot': 15,
              'nu': 3,
          },
          'numerics': {
              'q_correction_factor': 1.0,
          },
      },
  )
  geo = geometry.build_circular_geometry(
      n_rho=25,
      elongation_LCFS=1.72,
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
              5.20661629519172e-02,
              4.60988876154027e-01,
              1.26147053895562e00,
              2.43161302985008e00,
              3.94495426742821e00,
              5.77139204274191e00,
              7.88150913744143e00,
              1.02624902851883e01,
              1.29468903291818e01,
              1.60138019874064e01,
              1.95113851933952e01,
              2.33616710945446e01,
              2.73791502563530e01,
              3.13852483685790e01,
              3.52773174769370e01,
              3.90124464458183e01,
              4.25716551618680e01,
              4.59432154578932e01,
              4.91193048991180e01,
              5.20960097096705e01,
              5.48735276363101e01,
              5.74561065922613e01,
              5.98516821058499e01,
              6.20712100922494e01,
              6.41276789866938e01,
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(47.64848792277505),
      dr=geo.drho_norm,
  )
  psi_face_grad = np.array([
      0.0,
      10.22306783005273,
      20.01204157003973,
      29.25356227236148,
      37.8335309394533,
      45.66094438284257,
      52.75292736748793,
      59.52452869367077,
      67.11000109983858,
      76.67279145561574,
      87.43958014971919,
      96.25714752873601,
      100.43697904520883,
      100.15245280565131,
      97.30172770894852,
      93.37822422203246,
      88.98021790124204,
      84.28900740063128,
      79.40223603061902,
      74.41762026381333,
      69.43794816598867,
      64.56447389878174,
      59.88938783971438,
      55.48819965998799,
      51.41172236110965,
      47.64848792277505,
  ]).astype('float64')
  jtot = np.array([
      2.65553771917866e06,
      2.63458222735819e06,
      2.57127916870481e06,
      2.49314146142775e06,
      2.40050721598318e06,
      2.31654402077045e06,
      2.30295070171134e06,
      2.43327218739499e06,
      2.67541362889229e06,
      2.81376863491221e06,
      2.61677572372636e06,
      2.10372614588557e06,
      1.52490336359566e06,
      1.08634663861378e06,
      8.09871881607703e05,
      6.25330094597524e05,
      4.79703443455907e05,
      3.55359757835279e05,
      2.50048828655255e05,
      1.64394757045943e05,
      9.86238374628013e04,
      5.19943101413964e04,
      2.25943444867627e04,
      5.96407384624716e03,
      -2.16148333856769e03,
  ])
  s = np.array([
      -0.0,
      0.01061555243081,
      0.0471596469531,
      0.0842417481994,
      0.13088888090655,
      0.18238457877124,
      0.21451557789985,
      0.16603931954954,
      -0.00748384386686,
      -0.18762749511178,
      -0.13234887796843,
      0.23520003886811,
      0.75028418300229,
      1.19775706411421,
      1.48943252304723,
      1.67326712415659,
      1.82266186755645,
      1.97153957751977,
      2.12483352999861,
      2.27836487449329,
      2.42581488337183,
      2.56020627046558,
      2.67487034639264,
      2.76515993294247,
      2.83783886733272,
      3.09841399949699,
  ])
  return References(
      runtime_params=runtime_params,
      geometry_provider=geometry_provider_lib.ConstantGeometryProvider(geo),
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
      Ip_from_parameters=True,
  )


def chease_references_Ip_from_chease() -> References:  # pylint: disable=invalid-name
  """Reference values for CHEASE geometry where the Ip comes from the file."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  runtime_params = config_args.recursive_replace(
      runtime_params,
      **{
          'profile_conditions': {
              'Ip_tot': 15,
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
      813157.2571536204,
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
      -0.0,
      -0.03606779373088,
      -0.11809879887773,
      -0.14544622113962,
      -0.29119095449313,
      -0.4789507799013,
      -0.48703968044757,
      -0.29569374572961,
      0.05244002541098,
      0.42804329193027,
      0.71706284307321,
      0.9018555843838,
      1.03247014023787,
      1.15966789219429,
      1.28980722144808,
      1.42203542390446,
      1.55626269677731,
      1.70075987549412,
      1.85810051057465,
      2.03907899295674,
      2.31031334507313,
      2.57670435924605,
      2.31572983388489,
      1.79040411178413,
      1.46582319478329,
      1.25337716370375,
  ])
  return References(
      runtime_params=runtime_params,
      geometry_provider=geometry_provider_lib.ConstantGeometryProvider(geo),
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
      Ip_from_parameters=False,
  )


def chease_references_Ip_from_runtime_params() -> References:  # pylint: disable=invalid-name
  """Reference values for CHEASE geometry where the Ip comes from the config."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  runtime_params = config_args.recursive_replace(
      runtime_params,
      **{
          'profile_conditions': {
              'Ip_tot': 15,
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
      1036332.4408819571,
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
      -0.0,
      -0.03606779373088,
      -0.11809879887773,
      -0.14544622113962,
      -0.29119095449314,
      -0.4789507799013,
      -0.48703968044756,
      -0.29569374572963,
      0.05244002541104,
      0.42804329193025,
      0.7170628430731,
      0.90185558438396,
      1.03247014023779,
      1.15966789219413,
      1.28980722144824,
      1.42203542390441,
      1.55626269677737,
      1.70075987549401,
      1.85810051057471,
      2.03907899295714,
      2.31031334507276,
      2.57670435924617,
      2.31572983388459,
      1.79040411178401,
      1.46582319478391,
      1.2533771637032,
  ])
  return References(
      runtime_params=runtime_params,
      geometry_provider=geometry_provider_lib.ConstantGeometryProvider(geo),
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
      Ip_from_parameters=True,
  )


class ReferenceValueTest(parameterized.TestCase):
  """Unit using reference values from previous executions."""

  def setUp(self):
    super().setUp()
    torax.set_jax_precision()
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
