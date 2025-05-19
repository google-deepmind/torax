Tutorials
#########

Tutorials are available at |torax/tutorials/|. The tutorial is comprised of a
series of exercises. An additional notebook containing suggested solutions is
also available. The tutorial is tested for TORAX v1.0.0 and is not guaranteed to
work with later major versions due to API changes.

The tutorials are intended to be run in either a Jupyter or Colab notebook. It
is currently necessary to build your own notebook kernel in the same environment
where you installed TORAX.

First install the necessary dependencies. For PyPI installations:

.. code-block:: console

  pip install torax[tutorial]

For installations from the cloned source code, run the following from the TORAX
root directory:

.. code-block:: console

  pip install -e .[tutorial]

Then run the kernel:

.. code-block:: console

  jupyter notebook --no-browser

If successful, you should see a message like:

.. code-block:: console

  To access the server, open this file in a browser:
      file:///<PATH>
  Or copy and paste one of these URLs:
      http://localhost:8890/tree?token=<TOKEN>
      http://127.0.0.1:8890/tree?token=<TOKEN>

With ``<PATH>`` and ``<TOKEN>`` being environment and instance specific.

To access the notebook server, carry out the following steps for either a
Google Colab or Jupyter session, depending on your preference:

Google Colab
------------

For loading the tutorial notebook directly from GitHub, click the link below:

.. raw:: html

  <a target="_blank" href="https://colab.research.google.com/github/
  google-deepmind/torax/blob/main/torax/tutorials/
  torax_tutorial_exercises.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"
    alt="Open In Colab"/>
  </a>

And for the notebook containing the suggested solutions, click the link below:

.. raw:: html

  <a target="_blank" href="https://colab.research.google.com/github/
  google-deepmind/torax/blob/main/torax/tutorials/
  torax_tutorial_exercises_with_solutions.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"
    alt="Open In Colab"/>
  </a>

Carry out the following steps to connect to the notebook server:

1. Click on the arrow by the "Connect" button in the top right corner.
2. Choose "Connect to a local runtime", and paste the URL into the "Backend URL"
   field.

For loading the tutorial notebook from your cloned source code,
carry out the following steps:

1. Navigate your browser to https://colab.research.google.com/
2. Click on "File" in the upper left corner, then "Upload notebook", then
   "Browse".
3. Navigate to the ``torax/tutorials`` directory in the file explorer and upload
   the desired notebook.

Connect to the local runtime as described above.

Jupyter
-------

1. Copy one of the URLs into your browser.
2. Assuming that you have cloned the TORAX source code, navigate to the
   ``torax/tutorials`` directory in the file explorer.
3. Open the desired notebook.

The exercises focus on investigating the impact of heating and current drive
actuators on q-profile tailoring for a scenario inspired by the ITER hybrid
scenario. Further context is provided within the tutorial notebook itself.

To facilitate the tutorial, a baseline TORAX configuration has been provided.
Additionally, various helper functions for simulation config modification,
simulation execution, and plotting routines, are also all packaged with the
notebook. Instructions for using these routines are provided in the notebook
itself.
