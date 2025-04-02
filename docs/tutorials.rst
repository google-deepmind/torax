Tutorials
#########

Tutorial notebooks are WIP. First versions are available at
https://github.com/google-deepmind/torax/tree/main/torax/tutorials.

The tutorials are intended to be run in either a Jupyter or Colab notebook.

It is necessary to build your own notebook kernel in the same environment
where you installed TORAX. These tutorials are tested for TORAX v0.3.1 and are
not guaranteed to work with later versions due to API changes.

First install the necessary dependencies. Running from the torax root directory:

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

To access the notebook server, carry out the following steps for either a Jupyter
or Google Colab session, depending on your preference:

Jupyter
-------

1. Copy one of the URLs into your browser.
2. Navigate to the ``torax/tutorials`` directory in the file explorer.
3. Open the desired notebook.

Google Colab
------------

1. Navigate your browser to https://colab.research.google.com/
2. Click on the arrow by the "Connect" button in the top right corner.
3. Choose "Connect to a local runtime", and paste the URL into the "Backend URL"
   field.
4. Click on "File" in the upper left corner, then "Upload notebook", then "Browse".
5. Navigate to the ``torax/tutorials`` directory in the file explorer and upload
   the desired notebook.


The tutorial is comprised of a series of exercises. An additional notebook
containing suggested solutions is also available.

The exercises focus on investigating the impact of heating and current drive
actuators on q-profile tailoring for a scenario inspired by the ITER hybrid scenario.
Further context is provided within the tutorial notebook itself.

To facilitate the tutorial, a baseline TORAX configuration has been provided.
Additionally, various helper functions for simulation config modification,
simulation execution, and plotting routines, are also all packaged with the notebook.
Instructions for using these routines are provided in the notebook itself.
Some of these routines may soon be made available as part of the TORAX API.
