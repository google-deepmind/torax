.. _solver_config:

solver
######

Select and configure the ``Solver`` object, which evolves the PDE system by one
timestep. See :ref:`solver_details` for further details. The dictionary consists
of keys common to all solvers. Additional fields for parameters pertaining to a
specific solver are defined in the relevant section below.

.. torax_config:: torax._src.solver.config.LinearThetaMethodConfig
   :include-fields: solver_type

.. torax_config:: torax._src.solver.config.SolverConfigBase

linear
======

.. torax_config:: torax._src.solver.config.LinearThetaMethodConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: solver_type

newton_raphson
==============

.. _log_iterations:

.. torax_config:: torax._src.solver.config.NewtonRaphsonThetaMethodConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: solver_type

optimizer
=========

.. torax_config:: torax._src.solver.config.OptimizerThetaMethodConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: solver_type
