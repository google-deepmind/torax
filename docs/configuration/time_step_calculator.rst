.. _time_step_calculator_config:

time_step_calculator
####################

Configures the ``TimeStepCalculator`` object, which calculates ``dt`` at
every timestep. The dictionary consists of ``calculator_type`` and keys common
to all time step calculators. Additional fields for parameters pertaining to a
specific ``calculator_type`` are defined in the relevant section below.

.. torax_config:: torax._src.time_step_calculator.config.ChiTimeStepCalculatorConfig
   :include-fields: calculator_type

.. torax_config:: torax._src.time_step_calculator.config.TimeStepCalculatorConfigBase

chi
===

.. torax_config:: torax._src.time_step_calculator.config.ChiTimeStepCalculatorConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: calculator_type

fixed
=====

.. torax_config:: torax._src.time_step_calculator.config.FixedTimeStepCalculatorConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: calculator_type

from_previous_dt
================

.. torax_config:: torax._src.time_step_calculator.config.FromPreviousTimeStepCalculatorConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: calculator_type
