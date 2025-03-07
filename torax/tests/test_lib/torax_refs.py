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
from torax.config import build_runtime_params
from torax.config import config_args
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import runtime_params as sources_params
from torax.stepper import pydantic_model as stepper_pydantic_model
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
  psidot: np.ndarray
  jtot: np.ndarray
  q: np.ndarray
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
  return build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
      t=t,
      dynamic_runtime_params_slice_provider=build_runtime_params.DynamicRuntimeParamsSliceProvider(
          runtime_params,
          transport=transport_model_params.RuntimeParams(),
          sources=sources,
          stepper=stepper_pydantic_model.Stepper(),
          torax_mesh=geometry_provider.torax_mesh,
      ),
      geometry_provider=geometry_provider,
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
      },
  )
  geo = geometry_pydantic_model.CircularConfig(
      n_rho=25,
      elongation_LCFS=1.72,
      hires_fac=4,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  ).build_geometry()
  # ground truth values copied from example executions using
  # array.astype(str),which allows fully lossless reloading
  psi = fvm.cell_variable.CellVariable(
      value=jnp.array(
          np.array([
              5.20759356763185e-02,
              4.61075402494995e-01,
              1.26170728323729e00,
              2.43206820843406e00,
              3.94566757703537e00,
              5.77213040025801e00,
              7.88044279922951e00,
              1.02534529362164e01,
              1.29215098336351e01,
              1.59750627929530e01,
              1.94795940161767e01,
              2.33541697410208e01,
              2.73933377711206e01,
              3.14091156179732e01,
              3.53034698354459e01,
              3.90388127442209e01,
              4.25979583155311e01,
              4.59694615524991e01,
              4.91455160851201e01,
              5.21222016465174e01,
              5.48997101297067e01,
              5.74822852044825e01,
              5.98778595671550e01,
              6.20973874594807e01,
              6.41538565179961e01,
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(47.64848792277505),
      dr=geo.drho_norm,
  )
  psi_face_grad = np.array([
      0.0,
      10.2249866704669,
      20.01579701855746,
      29.25902312991926,
      37.83998421503257,
      45.66157058056604,
      52.70780997428761,
      59.32525342467125,
      66.70142243546908,
      76.33882398294575,
      87.61328058059274,
      96.86439312110346,
      100.97920075249486,
      100.3944461713158,
      97.3588554368157,
      93.38357271937525,
      88.97863928275598,
      84.28758092420061,
      79.40136331552488,
      74.41713903493188,
      69.43771207973252,
      64.56437686939474,
      59.8893590668137,
      55.48819730814234,
      51.4117264628835,
      47.64848792277505,
  ]).astype('float64')
  psidot = np.array([
      6.15796838096829e-02,
      7.67776861533454e-02,
      9.14645450891100e-02,
      1.05826180471346e-01,
      1.20797043388939e-01,
      1.51857197587599e-01,
      3.66055430824331e-01,
      1.60920226665232e00,
      5.35622979148619e00,
      1.01528185911838e01,
      1.04684759809395e01,
      5.86884004424684e00,
      1.87906989052661e00,
      4.66830085414812e-01,
      2.15885154977055e-01,
      1.83861837131887e-01,
      1.69158928431288e-01,
      1.51395781445053e-01,
      1.29766897536285e-01,
      1.04971917969482e-01,
      7.82838019851360e-02,
      5.16475815738970e-02,
      2.77752500731658e-02,
      9.90033116980977e-03,
      -1.40569237592991e-02,
  ])
  jtot = np.array([
      2.65603610444476e06,
      2.63507518532550e06,
      2.57173156499720e06,
      2.49325068732876e06,
      2.39844184311764e06,
      2.30679250085058e06,
      2.28072077110533e06,
      2.41165009601333e06,
      2.68616246795863e06,
      2.86383606047094e06,
      2.66405185162365e06,
      2.11004768810304e06,
      1.50210057505178e06,
      1.06527275121419e06,
      8.00985587848605e05,
      6.23310701928993e05,
      4.79501434000615e05,
      3.55403340091418e05,
      2.50094779129977e05,
      1.64425604025163e05,
      9.86423519234078e04,
      5.20041357205945e04,
      2.25986371083383e04,
      5.96528658619718e03,
      -2.16166743039443e03,
  ])
  q = np.array([
      0.65136284674552,
      0.65136284674552,
      0.66549200308491,
      0.68288435974473,
      0.70403585664969,
      0.72929778158408,
      0.75816199863277,
      0.78585816811502,
      0.79880472498816,
      0.78520449625846,
      0.76017886574672,
      0.75633510231278,
      0.79147108029917,
      0.86242114812994,
      0.95771945489906,
      1.06980964076381,
      1.19762252680817,
      1.34329397040352,
      1.50983774906477,
      1.70045978288947,
      1.91831678381418,
      2.16626740192572,
      2.44657621398095,
      2.76066019839069,
      3.10910068989888,
      3.49443220339156,
  ])
  s = np.array([
      -0.0,
      0.01061557184301,
      0.04716064616925,
      0.08426669061692,
      0.13122491602167,
      0.1848822350657,
      0.22446393489629,
      0.18458330229119,
      -0.0033848034437,
      -0.22475896767388,
      -0.18476785956073,
      0.21635274198114,
      0.77234423397151,
      1.22946678897025,
      1.50693701200676,
      1.67820617110573,
      1.82332806784304,
      1.9714865968474,
      2.12473729256083,
      2.27829069028721,
      2.4257636557766,
      2.56017437523365,
      2.67485320711614,
      2.76515311932521,
      2.83783799296473,
      3.09841616536076,
  ])
  return References(
      runtime_params=runtime_params,
      geometry_provider=geometry_provider_lib.ConstantGeometryProvider(geo),
      psi=psi,
      psi_face_grad=psi_face_grad,
      psidot=psidot,
      jtot=jtot,
      q=q,
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
      },
  )
  geo = geometry_pydantic_model.CheaseConfig(
      geometry_dir=_GEO_DIRECTORY,
      geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      n_rho=25,
      Ip_from_parameters=False,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  ).build_geometry()
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
  psidot = np.array([
      0.01900393199299,
      0.02502967488295,
      0.03137343719743,
      0.03872260497412,
      0.05304578115468,
      0.07204449940502,
      0.11982232309585,
      0.33599133626923,
      0.97622066869252,
      1.79293945666743,
      1.83253787252562,
      1.03282506068381,
      0.34893930670678,
      0.11119916091885,
      0.07110748084107,
      0.0687649756332,
      0.07024246260073,
      0.07270482339355,
      0.07725935134098,
      0.08187661259011,
      0.08383698677169,
      0.11866443718669,
      0.21390251275037,
      0.40289033343704,
      0.61986455206849,
  ])
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
  q = np.array([
      1.74778489687499,
      1.74778489687499,
      1.63019017120101,
      1.55135548433531,
      1.47944703389477,
      1.34587749007678,
      1.22208416710003,
      1.14174321035495,
      1.12073724167867,
      1.15719843686828,
      1.23446022617573,
      1.33693108455852,
      1.45471199796885,
      1.58807953262815,
      1.73889524213944,
      1.90938587558384,
      2.10182786843057,
      2.31924672085511,
      2.56737489962044,
      2.85095694625504,
      3.18252465224016,
      3.5949796873733,
      4.06580137921761,
      4.41721011791634,
      4.74589637288284,
      4.98383229828587,
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
      psidot=psidot,
      jtot=jtot,
      q=q,
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
      },
  )
  geo = geometry_pydantic_model.CheaseConfig(
      geometry_dir=_GEO_DIRECTORY,
      geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      n_rho=25,
      Ip_from_parameters=True,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  ).build_geometry()
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
  psidot = np.array([
      0.02405831545108,
      0.03169056039557,
      0.03972149390638,
      0.04902499960571,
      0.06715742163186,
      0.09120822167296,
      0.15169177157271,
      0.42534900161173,
      1.2358354798277,
      2.26973763607176,
      2.31986038070078,
      1.30748145802947,
      0.44173257747467,
      0.14077087044304,
      0.09001802852518,
      0.08705341319902,
      0.08892506417214,
      0.09204402879046,
      0.0978123708192,
      0.10366110216499,
      0.10614720189507,
      0.15024986030514,
      0.27085114402416,
      0.51017615337295,
      0.78493833987809,
  ])
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
  q = np.array([
      1.37139774532955,
      1.37139774532955,
      1.27912715646002,
      1.21726959491758,
      1.16084669815229,
      1.05604148352651,
      0.95890717122028,
      0.89586771645968,
      0.87938540325705,
      0.90799464514058,
      0.9686180341204,
      1.04902169500535,
      1.14143837590391,
      1.24608508423676,
      1.36442248625569,
      1.49819779849434,
      1.64919722386333,
      1.81979471817761,
      2.01448825599212,
      2.23700063727324,
      2.4971649202036,
      2.82079736847724,
      3.19022715803668,
      3.46595968828215,
      3.72386304343085,
      3.91055923940663,
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
      psidot=psidot,
      jtot=jtot,
      q=q,
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
