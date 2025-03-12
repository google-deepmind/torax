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
import sys
import torax
torax_path, = torax.__path__ # Not sure why this is a length 1 list
# run_simulation_main.py is in the repo root, which is the parent directory
# of the actual module
torax_repo_path = os.path.abspath(os.path.join(torax_path, os.pardir))
import sys
sys.path.append(torax_repo_path)
import run_simulation_main


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
    run_simulation_main_path = os.path.join(torax_repo_path,
    'run_simulation_main.py')
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

    # Number of seconds we expect the cache to save on the second run
    # This rule is likely to need adjusting to support more machine types
    # and more CI environments, or possibly may need adjusting for
    # flakiness (in initial testing of this rule it passed 100 / 100 runs)
    # so be suspicious if it becomes highly flaky without a good reason.
    thresh = 8.53

    speedup = t0 - t1

    success = speedup > thresh

    if not success:
      print('Cache did not significantly accelerate second run.')
      print('Debugging info:')
      print('Args to subprocess:', subprocess_args)
      print('Output of first call:')
      print(out0)
      print('Output of second call:')
      print(out1)

      # Cache did not significantly accelerate second run
      contents = os.listdir(cache)
      # If the cache is empty, it is because we are not writing to it at all
      self.assertNotEmpty(contents)
      print('Contents of cache:')
      for path in contents:
        print('\t', path)
      # Debugging information:
      # As of 2024-07-24 the cache should contain
      # jit__calc_coeffs_full-<hash>
      # jit_theta_method_block_jacobian-<hash>
      # jit_theta_method_block_residual-<hash>
      # If any of these does not appear, grep out0 for these names and look
      # for a reason it was not written. Make sure errors are disabled in
      # torax.jax_utils because errors use callbacks and callbacks cannot
      # be serialized to the persistent cache.
      # You can also run run_simulation_main on the command line twice,
      # starting with an
      # empty cache directory and see if it gets a cache hit.
      # If it does get a cache hit, inspect, the directory to see if the
      # above list of objects that should be in the cache is still up to
      # date.

      msg = io.StringIO()

      eprint = functools.partial(print, file=msg)

      eprint('Cache did not successfully accelerate run_simulation main.')
      eprint('First run time: ', t0)
      eprint('Second run time: ', t1)

      raise AssertionError(msg.getvalue())


if __name__ == '__main__':
  absltest.main()
