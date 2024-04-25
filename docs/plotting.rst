Plotting simulations
####################

To plot the output of a single simulation, run the following command from the TORAX
root directory:

.. code-block:: console

  python3 plotting/plotruns.py --outfile <full_path_to_simulation_output>

Alternatively, ``plotting/plotruns.py`` can be replaced by the relative path and the
command run from anywhere. The command will plot the following outputs:

* Ion and electron heat conductivity
* Ion and electron temperature
* Electron density
* Total, Ohmic, bootstrap, and external current
* q-profile
* Magnetic shear

A slider allows to scroll through the output of all simulation timesteps.

To plot the output of two simulations on top of each other, run the following command:

.. code-block:: console

  python3 plotting/plotruns.py --outfile <full_path_to_simulation_output1> \\
   <full_path_to_simulation_output2>

Upgrades to TORAX visualization capabilities are planned.
