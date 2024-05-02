.. _roadmap:

Development Roadmap
###################

Short term development plans include:

* Time dependent geometry
* More flexible functional forms for initial and prescribed conditions
* Implementation of forward sensitivity calculations w.r.t. control inputs and parameters
* Implementation of persistent compilation cache for CPU
* Extended visualisation

Longer term desired features include:

* Sawtooth model (Porcelli + reconnection)
* Neoclassical tearing modes (modified Rutherford equation)
* Radiation sinks

  * Cyclotron radiation
  * Bremsstrahlung
  * Line radiation

* Neoclassical transport + multi-ion transport, with a focus on heavy impurities
* IMAS coupling
* Stationary-state solver
* Momentum transport

Contributions in line with the roadmap are welcome. In particular, TORAX is envisaged
as a natural framework for coupling of various ML-surrogates of physics models.
These could include surrogates for turbulent transport, neoclassical transport, heat
and particle sources, line radiation, pedestal physics, and core-edge integration, MHD, among others.
