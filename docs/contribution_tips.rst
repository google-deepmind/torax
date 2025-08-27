.. _contribution_tips:

Contribution tips
#################

Installing from source
======================

Create a code directory where you will install the virtual env and other TORAX
dependencies.

```shell
mkdir /path/to/torax_dir && cd "$_"
```
Where `/path/to/torax_dir` should be replaced by a path of your choice.

Create a TORAX virtual env.

```shell
python3 -m venv toraxvenv
source toraxvenv/bin/activate
```

Download and install the TORAX codebase via http:

```shell
git clone https://github.com/google-deepmind/torax.git
```
or ssh (ensure that you have the appropriate SSH key uploaded to github).

```shell
git clone git@github.com:google-deepmind/torax.git
```
Enter the TORAX directory and pip install the dependencies.

```shell
cd torax; pip install -e .
```

If you want to install with the dev dependencies (useful for running `pytest`
and installing `pyink` for lint checking), then run with the `[dev]`:

```shell
cd torax; pip install -e .[dev]
```

Optional: Install additional GPU support for JAX if your machine has a GPU:
https://jax.readthedocs.io/en/latest/installation.html#supported-platforms


Code reviews
============

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
`[GitHub Help] <https://help.github.com/articles/about-pull-requests/>`_ for more
information on using pull requests.

Documentation
=============

If appropriate for the change you are making, please update the documentation
as part of your contribution.

TORAX's documentation is written in `[reStructuredText] <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

All documentation is found in the torax/docs directory, in .rst files. Please either
modify the existing files or create new files and links as appropriate.

Once you have written your documentation, you can stage it to see how it looks
before you commit it to the repository. To stage your documentation, run the
following command from the docs folder in the TORAX repository:

.. code-block:: console

  make html

This will generate the documentation in the `docs/_build/html` directory. You can then view
the documentation by opening the `index.html` file in your browser.
Run ``pwd`` to find the path to the `docs/_build/html` directory and then point your browser
to `file:///path/to/torax/docs/_build/html/index.html`.

The staged documentation is cleaned up from your local repository by running the command:

.. code-block:: console

  make clean

Following the submission of your pull request, the documentation will be
automatically regenerated and published to TORAX's readthedocs server.

Testing
=======

TORAX's tests are written in the `[pytest] <https://docs.pytest.org/en/stable/>`_ framework

While all tests are run automatically on GitHub when you push a pull request, it
is a good idea to run them locally while you are developing your feature. Ensure
that you have installed the required developer dependencies (including pytest)
by having run the following command from the TORAX root directory:

.. code-block:: console

  pip install -e .[dev]

To run all tests, run the following command from the TORAX root directory:

.. code-block:: console

  pytest -n <num_workers>

with `<num_workers>` being the number of processes that you want to use in parallel.

It is recommended to run tests with the environment variable ``TORAX_ERRORS_ENABLED=True`` to
enable full test coverage. However, it is then recommended to revert back to ``TORAX_ERRORS_ENABLED=False``
when running TORAX in production mode, to enable the persistent JAX cache.

To run a specific test, run the following command from the TORAX root directory,
in this case running all the geometry tests.

.. code-block:: console

  pytest torax/_src/geometry/tests/geometry_test.py

Further filtering is possible, for example running only the ``test_face_to_cell`` test (in geometry.py):

.. code-block:: console

  pytest -k face_to_cell

Which runs any test containing the string expression ``face_to_cell``.

Where appropiate, please add tests for your changes.

An important class of test is the sim test. These are integration tests running
the configs in the ``torax/tests/test_data/`` directory, and comparing to the ground-truth
``.nc`` TORAX outputs found in the same directory. Sim tests can be triggered separately
by a command (from the TORAX root directory) such as:

.. code-block:: console

  pytest -n <num_workers> torax/tests/sim_test.py

If any sim tests fail, they write their output to the ``/tmp/torax_failed_sim_test_outputs/<test_name>.nc``.
This is useful for debugging, and also to stage new output files for replacing the ground-truth files,
if you expect that your change to the code produces different outputs.

To compare the absolute and relative differences between the failed sim tests
to the ground-truth files, run the following command from the TORAX root directory:

.. code-block:: console

  python3 torax/tests/scripts/compare_sim_tests.py

These command has the optional flag ``--failed_test_output_dir <dir>`` which
takes a directory containing the failed test outputs, instead of the default
directory ``/tmp/torax_failed_sim_test_outputs``.

It is sometimes useful to plot the difference between the ground-truth and a
failed TORAX sim test, either for debugging or to verify that the magnitude of
difference is as expected. To do this, run the following command from the root of
the TORAX repository. Using ``test_qlknnheat`` as an example:

.. code-block:: console

  plot_torax --outfile torax/tests/test_data/test_qlknnheat.nc /tmp/torax_failed_sim_test_outputs/test_qlknnheat.nc

If it is deemed that the new outputs should replace the ground-truth files,
they can be copied over using the following command, again with this example working
when run from the TORAX repository root:

.. code-block:: console

  python3 torax/tests/scripts/copy_sim_tests.py

Where we also have the optional flag ``--failed_test_output_dir <dir>`` which
takes a directory containing the failed test outputs, instead of the default
directory ``/tmp/torax_failed_sim_test_outputs``.

Finally, there are use-cases where it is desirable to rerun all the sim tests,
even if the tests are passing. An example is when the output API changes and we
wish to keep all the test ``.nc`` files up-to-date. In this case, run the following
command from the TORAX root directory:

.. code-block:: console

  python3 torax/tests/scripts/run_and_save_all_benchmarks.py

This script has the following optional flags:

* ``--output_dir`` (default ``/tmp/torax_sim_outputs``): directory where to save the outputs
* ``--num_proc`` (default ``16``): number of processes to use

The ``compare_sim_tests.py`` can be used for sanity checking the outputs, and the
``copy_sim_tests.py`` can be used to replace the ground-truth files. Note that the
``--failed_test_output_dir`` flag in the compare and copy scripts needs to be set
to the same output directory as the ``run_and_save_all_benchmarks.py`` script.

Regenerating Unit Test References
================================

Some unit tests (particularly for physics and geometry calculations) rely on
pre-calculated reference values stored in torax/_src/test_utils/references.json.
If you make a change that intentionally and correctly alters these values, you
will need to regenerate this reference file.
To do this, a dedicated script is provided.

Regenerating All Reference Cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To regenerate all reference cases and overwrite the existing JSON file, run the
following command from the TORAX root directory:

.. code-block:: console

  python3 torax/tests/scripts/regenerate_torax_refs.py --write_to_file

For printing a summary of the regenerated values to stdout for quick inspection,
add a ``--print_summary`` flag. For a dry-run with no write, remove the
``--write_to_file`` flag.

Regenerating a Specific Case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you only need to update a specific reference case (e.g.,
circular_references), you can specify it with the --case flag.

.. code-block:: console

  python3 torax/tests/scripts/regenerate_torax_refs.py --case=circular_references --write_to_file

.. important::
  When making changes to the output structure, e.g. adding fields,
  a subset of the sim tests will fail. To pass these specific tests, it is
  required to update ``implicit.nc``, ``test_changing_config_before.nc``, and
  ``test_changing_config_after.nc``. However, the recommended workflow when
  changing   output API is to run the ``run_and_save_all_benchmarks.py`` script,
  which also updates the aforementioned files. When doing so, it is further
  strongly recommended to afterwards run the ``compare_sim_tests.py`` script to
  verify that the changes to the ground-truth files are as expected. For pure
  output API changes, these should be zero. Results of ``compare_sim_tests.py``
  should be shared in the pull request discussion.


