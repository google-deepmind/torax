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

_GEO_DIRECTORY = 'torax/data/third_party/geo'

# It's best to import the parent `torax` package because that has the
# __init__ file that configures jax to float64
fvm = torax.fvm
geometry = torax.geometry


@chex.dataclass(frozen=True)
class References:
  """Collection of reference values useful for unit tests."""

  runtime_params: general_runtime_params.GeneralRuntimeParams
  geo: geometry.Geometry
  psi: fvm.cell_variable.CellVariable
  psi_face_grad: np.ndarray
  jtot: np.ndarray
  s: np.ndarray


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
      2670452.904550778,
      2637520.0133290826,
      2577444.835299116,
      2501381.1883701743,
      2405344.399498295,
      2300636.482210432,
      2239714.123785538,
      2342164.9833642156,
      2665520.478746461,
      2953064.3504205365,
      2787290.0891781836,
      2157368.9961162456,
      1482107.219508777,
      1042774.4183074178,
      796734.7677713615,
      627485.7315059016,
      484728.14844338456,
      359907.1632643329,
      253687.07631308853,
      167140.5317040243,
      100554.82338511273,
      53218.37217192466,
      23262.50283731702,
      5798.103873077084,
      -3709.7562407235314,
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
      geo=geo,
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
  geo = geometry.build_geometry_from_chease(
      runtime_params=runtime_params,
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
              '0.028269200893290054',
              '0.25830070956630535',
              '0.7515864371427805',
              '1.5290377293407753',
              '2.6160416591477507',
              '4.109776781076197',
              '6.083267597177049',
              '8.548498997963945',
              '11.418026005498554',
              '14.54591564529029',
              '17.802294247322177',
              '21.11023166707459',
              '24.426030866786014',
              '27.71745280935002',
              '30.954224195485104',
              '34.112857792036955',
              '37.173258956533324',
              '40.12077464958477',
              '42.93919885653815',
              '45.61832569488219',
              '48.144560261511515',
              '50.492447775755906',
              '52.668083031356424',
              '54.76130159148528',
              '56.79525122154844',
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(50.417485084359726),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      '0.0',
      '5.750787716825383',
      '12.332143189411878',
      '19.43628230494987',
      '27.175098245174382',
      '37.34337804821115',
      '49.33727040252131',
      '61.63078501967238',
      '71.73817518836523',
      '78.19724099479339',
      '81.40946505079718',
      '82.69843549381032',
      '82.89497999278561',
      '82.28554856410017',
      '80.91928465337705',
      '78.96583991379629',
      '76.51002911240923',
      '73.68789232628608',
      '70.46060517383452',
      '66.97817095860098',
      '63.15586416573318',
      '58.69718785610978',
      '54.39088139001296',
      '52.330464003221344',
      '50.84874075157906',
      '50.417485084359726',
  ]).astype('float64')
  jtot = np.array([
      795930.3850287468,
      840806.067226453,
      907353.1392477998,
      997244.126228929,
      1147270.435920407,
      1299395.4801796344,
      1355790.843330903,
      1271026.4640183216,
      1082415.0484984242,
      883539.5778949445,
      735591.8600635966,
      636536.6948867801,
      560950.5390205022,
      495140.9948666199,
      438029.20715456794,
      390516.3940455103,
      349981.3013816164,
      315240.68778055167,
      286627.24447538453,
      257638.89564892004,
      242652.46242764295,
      282856.95091316843,
      395758.1703319502,
      521038.63194130256,
      579208.5082904168,
  ])
  s = np.array([
      -0.07221339029182716,
      -0.07221339029182716,
      -0.10978697094216668,
      -0.14556469833048682,
      -0.3182545587996119,
      -0.48410241240691904,
      -0.4777827504007501,
      -0.27329938411328436,
      0.07676160004898383,
      0.4477451568703172,
      0.7339849318492895,
      0.9180790330929585,
      1.056015149635723,
      1.1952782347275002,
      1.3428889005376405,
      1.4969801606698092,
      1.6477614361846962,
      1.822310073007422,
      2.0162016336630817,
      2.2346589515146733,
      2.570972907161161,
      2.897484271282114,
      2.626089343552705,
      2.078461443212857,
      1.7481696293083797,
      1.8139935034456611,
  ])
  return References(
      runtime_params=runtime_params,
      geo=geo,
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
  geo = geometry.build_geometry_from_chease(
      runtime_params=runtime_params,
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
              '0.03602782820394983',
              '0.32919266534421404',
              '0.9578631932332422',
              '1.9486899837739167',
              '3.334027722464634',
              '5.2377261169890375',
              '7.752851618871569',
              '10.894678433354123',
              '14.551761859387222',
              '18.538116868455745',
              '22.688225295074556',
              '26.904043121624614',
              '31.12988043400343',
              '35.324650025865274',
              '39.449769935526916',
              '43.475306738118505',
              '47.3756507133815',
              '51.13212722004177',
              '54.72408241952786',
              '58.138509372479355',
              '61.35808198484154',
              '64.35035928071211',
              '67.12310880126317',
              '69.79082194116809',
              '72.38299948888208',
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(64.25482269382654),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      '0.0',
      '7.329120928506605',
      '15.716763197225703',
      '24.77066976351686',
      '34.63344346726794',
      '47.59245986311008',
      '62.87813754706328',
      '78.54567036206386',
      '91.42708565082746',
      '99.65887522671308',
      '103.75271066547027',
      '105.39544566375145',
      '105.64593280947037',
      '104.86923979654614',
      '103.12799774154104',
      '100.63842006478971',
      '97.50859938157497',
      '93.91191266650658',
      '89.79887998715235',
      '85.36067382378735',
      '80.48931530905463',
      '74.8069323967643',
      '69.31873801377648',
      '66.69282849762297',
      '64.80443869284969',
      '64.25482269382654',
  ]).astype('float64')
  jtot = np.array([
      1014377.5652648797,
      1071569.6088198768,
      1156380.8663899498,
      1270942.8962210915,
      1462144.6967969288,
      1656021.2404169012,
      1727894.6005017115,
      1619866.2021321978,
      1379489.41534911,
      1126031.5508722072,
      937478.8223633758,
      811237.458584632,
      714906.2942044714,
      631034.9828113304,
      558248.5717671211,
      497695.6231839255,
      446035.4662435689,
      401760.1128919223,
      365293.5631789182,
      328349.2131996632,
      309249.6764449908,
      360488.49319299735,
      504376.03188187175,
      664040.3593317664,
      738175.2568713971,
  ])
  s = np.array([
      -0.07221339029182716,
      -0.07221339029182716,
      -0.10978697094216645,
      -0.1455646983304877,
      -0.3182545587996117,
      -0.4841024124069177,
      -0.47778275040075163,
      -0.2732993841132834,
      0.07676160004898405,
      0.447745156870319,
      0.7339849318492886,
      0.9180790330929495,
      1.0560151496357255,
      1.1952782347274975,
      1.3428889005376485,
      1.4969801606698174,
      1.647761436184692,
      1.822310073007411,
      2.0162016336630986,
      2.2346589515146715,
      2.570972907161099,
      2.8974842712821975,
      2.626089343552712,
      2.0784614432128303,
      1.7481696293084164,
      1.813993503445698,
  ])
  return References(
      runtime_params=runtime_params,
      geo=geo,
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
