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

_GEO_DIRECTORY = 'torax/data/third_party/geo'

# It's best to import the parent `torax` package because that has the
# __init__ file that configures jax to float64
config_lib = torax.config
fvm = torax.fvm
geometry = torax.geometry


@chex.dataclass(frozen=True)
class References:
  """Collection of reference values useful for unit tests."""

  config: config_lib.Config
  geo: geometry.Geometry
  psi: fvm.CellVariable
  psi_face_grad: np.ndarray
  jtot: np.ndarray
  s: np.ndarray


def circular_references() -> References:
  """Reference values for circular geometry."""
  # Hard-code the parameters relevant to the tests, so the reference values
  # will stay valid even if we change the Config constructor defaults
  config = config_lib.Config()
  config = config_lib.recursive_replace(
      config,
      **{
          'nr': 25,
          'Ip': 15,
          'q_correction_factor': 1.0,
          'nu': 3,
          'fext': 0.2,
          'wext': 0.05,
          'rext': 0.4,
      },
  )
  geo = geometry.build_circular_geometry(
      config=config,
      kappa=1.72,
      hires_fac=4,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  )
  # ground truth values copied from example executions using
  # array.astype(str),which allows fully lossless reloading
  psi = fvm.CellVariable(
      value=jnp.array(
          np.array([
              '0.05207731132547635',
              '0.4611644651838215',
              '1.262368686043099',
              '2.4345422957830465',
              '3.9522412798126245',
              '5.786153673325165',
              '7.904765846151847',
              '10.28484777914357',
              '12.95018071075735',
              '16.014437562460827',
              '19.590773702521638',
              '23.594685936482705',
              '27.772054134965423',
              '31.91761917182856',
              '35.944973231606774',
              '39.82484086119695',
              '43.5413794838048',
              '47.082299974042456',
              '50.438503337233755',
              '53.60464534005333',
              '56.579479705895324',
              '59.36590219377724',
              '61.970681772505465',
              '64.40386169750948',
              '66.67780971430767',
          ]).astype('float64')
      ),
      right_face_grad_constraint=jnp.array(53.182574789531735),
      dr=geo.dr_norm,
  )
  psi_face_grad = np.array([
      '0.0',
      '10.227178846458628',
      '20.030105521481936',
      '29.30434024349869',
      '37.94247460073945',
      '45.847809837813514',
      '52.96530432066704',
      '59.50204832479309',
      '66.63332329034448',
      '76.60642129258694',
      '89.40840350152025',
      '100.09780584902667',
      '104.43420496206794',
      '103.6391259215784',
      '100.68385149445538',
      '96.99669073975431',
      '92.91346556519642',
      '88.52301225594132',
      '83.90508407978245',
      '79.1535500704894',
      '74.37085914604981',
      '69.66056219704785',
      '65.11948946820567',
      '60.82949812510048',
      '56.84870041995467',
      '53.182574789531735',
  ]).astype('float64')
  jtot = np.array([
      2670238.90049012,
      2637300.7366563296,
      2577217.2145503103,
      2501145.8658131263,
      2405102.406655332,
      2300392.631913307,
      2239487.185200981,
      2341982.7457549125,
      2665348.5368826697,
      2952759.369088318,
      2786768.044807202,
      2156770.23717709,
      1481623.4662646332,
      1042435.8989011978,
      796473.3924649016,
      627256.298345361,
      484520.1557493787,
      359720.9523943518,
      253523.94449627143,
      167000.98829914053,
      100438.27451400229,
      53122.95708559138,
      23185.001091658065,
      6220.315580706229,
      -2275.8890473756223,
  ])
  s = np.array([
      0.020741407665038847,
      0.020741407665038847,
      0.04662968535721597,
      0.0802947160835128,
      0.12207872200306222,
      0.17055280728776842,
      0.21149520509496259,
      0.1811645967831043,
      -0.024519388744436817,
      -0.30671127978158813,
      -0.2824813106438732,
      0.1558181647209806,
      0.7065223014230788,
      1.0880157225214704,
      1.279111833764353,
      1.3919393753546678,
      1.494886824594276,
      1.602666157682944,
      1.7135497082882751,
      1.8229793630080255,
      1.9255939858458673,
      2.0156686089644733,
      2.087851925147558,
      2.1385247546658452,
      2.171367486759362,
      2.3139498913449468,
  ])
  return References(
      config=config,
      geo=geo,
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
  )


def chease_references_Ip_from_chease() -> References:  # pylint: disable=invalid-name
  """Reference values for CHEASE geometry where the Ip comes from the file."""
  config = config_lib.Config()
  config = config_lib.recursive_replace(
      config,
      **{
          'nr': 25,
          'Ip': 15,
          'q_correction_factor': 1.0,
          'nu': 3,
          'fext': 0.2,
          'wext': 0.05,
          'rext': 0.4,
      },
  )
  geo = geometry.build_chease_geometry(
      config=config,
      geometry_dir=_GEO_DIRECTORY,
      geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      Ip_from_parameters=False,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  )
  # ground truth values copied from an example PINT execution using
  # array.astype(str),which allows fully lossless reloading
  psi = fvm.CellVariable(
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
      config=config,
      geo=geo,
      psi=psi,
      psi_face_grad=psi_face_grad,
      jtot=jtot,
      s=s,
  )


def chease_references_Ip_from_config() -> References:  # pylint: disable=invalid-name
  """Reference values for CHEASE geometry where the Ip comes from the config."""
  config = config_lib.Config()
  config = config_lib.recursive_replace(
      config,
      **{
          'nr': 25,
          'Ip': 15,
          'q_correction_factor': 1.0,
          'nu': 3,
          'fext': 0.2,
          'wext': 0.05,
          'rext': 0.4,
      },
  )
  geo = geometry.build_chease_geometry(
      config=config,
      geometry_dir=_GEO_DIRECTORY,
      geometry_file='ITER_hybrid_citrin_equil_cheasedata.mat2cols',
      Ip_from_parameters=True,
      Rmaj=6.2,
      Rmin=2.0,
      B0=5.3,
  )
  # ground truth values copied from an example executions using
  # array.astype(str),which allows fully lossless reloading
  psi = fvm.CellVariable(
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
      config=config,
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
        chease_references_Ip_from_config()
    )
    # pylint: enable=invalid-name

if __name__ == '__main__':
  absltest.main()
