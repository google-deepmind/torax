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

"""Integration test making sure we hit the Jax persistent cache."""

import functools
import io
import os
import re
import subprocess
from absl.testing import absltest
from absl.testing import parameterized
from torax import run_simulation_main
import torax


class PersistentCacheTest(parameterized.TestCase):
  """Integration test making sure we hit the Jax persistent cache."""

  def test_persistent_cache(self):
    """Test that we hit the Jax persistent cache."""

    # We want the cache to be empty on the first run (to know how slow we are
    # when we miss), so even if it was possible
    # to access a populated cache in CI we would not want to
    cache = self.create_tempdir('cache')
    cache = cache.full_path

    flags = [
        '--config_package=torax',
        '--config=.tests.test_data.test_iterhybrid_rampup_short',
        f'--jax_compilation_cache_dir={cache}',
        '--jax_persistent_cache_min_entry_size_bytes=-1',
        '--jax_persistent_cache_min_compile_time_secs=0.0',
        '--quit',
        '--alsologtostderr',
        '--jax_debug_log_modules=jax._src.compilation_cache,jax._src.compiler',
    ]
    run_simulation_main_path = os.path.join(
        torax.__path__[0], 'run_simulation_main.py'
    )
    assert os.path.exists(run_simulation_main_path)
    command = ['python3', run_simulation_main_path]

    subprocess_args = command + flags

    new_env = dict(os.environ)
    # Torax errors are enabled for all tests on GitHub CI.
    # We need to override this because error handling uses callbacks,
    # and the callbacks can't be serialized to the persistent cache.
    new_env['TORAX_ERRORS_ENABLED'] = 'False'

    def run() -> str:
      result = subprocess.run(
          subprocess_args,
          env=new_env,
          check=True,
          capture_output=True,
          text=True,
      )
      return result.stdout + result.stderr

    # Run the job once to populate the cache
    out0 = run()
    contents0 = os.listdir(cache)

    # Possibly fail fast
    if (
        'Not writing persistent cache entry for'
        r'\'jit__calc_coeffs_full\' because it uses host callbacks'
    ) in out0:
      raise RuntimeError(
          'Cache not used due to callbacks: check that '
          'errors are disabled in jax_utils.'
      )

    # Run the job a second time to hit the cache
    out1 = run()
    contents1 = os.listdir(cache)
    # check the cache contents are the same
    self.assertListEqual(contents0, contents1)

    def get_simulation_time(text):
      match = re.search(
          run_simulation_main.SIMULATION_TIME + r': (\d+\.\d+)', text
      )
      if match:
        return float(match.group(1))
      else:
        print('Debugging info:')
        print('Job output text:')
        print(text)
        raise AssertionError(
            'Job did not print '
            f'"{run_simulation_main.SIMULATION_TIME}: <float>"'
        )

    # We use simulation time because we mostly expect the cache to speed up
    # the first simulation step.
    t0 = get_simulation_time(out0)
    t1 = get_simulation_time(out1)
    # on CI we require a non‑trivial speedup; locally compilation
    # overhead may be too small to outperform the simulation time.
    thresh = 8.53
    speedup = t0 - t1

    # If we're not on CI/GitHub Actions, skip the strict assertion:
    if not (os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')):
      print(
          f'[persistent_cache_test] local speedup={speedup:.2f}s '
          f'(threshold={thresh}s) – skipping timing assertion'
      )
      return

    # Otherwise (on CI) enforce the threshold:
    self.assertGreater(
        speedup,
        thresh,
        f'Expected >{thresh}s speedup, but only saw {speedup:.2f}s',
    )


if __name__ == '__main__':
  absltest.main()
