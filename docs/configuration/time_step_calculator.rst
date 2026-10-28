.. _time_step_calculator_config:

time_step_calculator
####################

``calculator_type`` (str [default = 'chi'])
  The name of the ``time_step_calculator``, a method which calculates ``dt`` at
  every timestep. Three methods are currently available:

* ``'fixed'``
    ``dt`` is equal to ``fixed_dt`` defined in :ref:`numerics_dataclass`.
    If the Newton-Raphson solver is being used and ``adaptive_dt==True`` in the
    ``numerics`` config dict, then in practice some steps may have lower ``dt``
    if the solver needed to backtrack.

* ``'chi'``
    adaptive dt method, where ``dt`` is a multiple of a base dt inspired by the
    explicit stability limit for parabolic PDEs:
    :math:`dt_{base}=\frac{dx^2}{2\chi}`, where :math:`dx` is the grid
    resolution  and :math:`\chi=max(\chi_i, \chi_e)`.
    ``dt=chi_timestep_prefactor * dt_base``, where ``chi_timestep_prefactor`` is
    defined in :ref:`numerics_dataclass`, and can be significantly larger than
    unity for implicit solvers. Scaling the timestep to be :math:`\propto \chi`
    helps protect against traversing through fast transients, if there is a
    desire for them to be fully resolved.

* ``'from_previous_dt'``
    ``dt`` is equal to the previous time step duration multiplied by the
    ``dt_reduction_factor`` defined in :ref:`numerics_dataclass`. This can speed
     up simulations that take many dt reduction steps, as the dt is initialized
     to a smaller value, rather than starting from max_dt every time. It can
     also speed up simulations where a small dt is required for a short section,
     but then a larger dt is be appropriate for the remainder of the simulation.

``tolerance`` (float [default = 1e-7])
  The tolerance within the final time for which the simulation will be
  considered done.
