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
from absl.testing import absltest
from absl.testing import parameterized
from torax._src.transport_model import tglfnn_ukaea_transport_model


class TglfnnUkaeaTransportModelTest(parameterized.TestCase):

  def test_hash_and_eq_same(self):

    model1 = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    model2 = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )

    self.assertEqual(hash(model1), hash(model2))
    self.assertEqual(model1, model2)


if __name__ == '__main__':
  absltest.main()
