TORAX: Tokamak transport simulation in JAX
==========================================

TORAX is a differentiable tokamak core transport simulator aimed for fast and
accurate forward modelling, pulse-design, trajectory optimization, and controller
design workflows. TORAX is written in Python using the JAX_ library.

.. grid:: 1 1 3 3
   :gutter: 4
   :class-container: torax-pillars-grid

   .. grid-item-card:: Flexible
      :class-card: sd-border-0 torax-pillar-card
      :shadow: None

      Python facilitates coupling within various workflows and to additional
      physics models. Easy to install and JAX can seamlessly execute
      on multiple backends including CPU and GPU.

   .. grid-item-card:: Fast and auto-differentiable
      :class-card: sd-border-0 torax-pillar-card
      :shadow: None

      JAX provides just-in-time compilation for fast runtimes. JAX auto-differentiability
      enables gradient-based nonlinear PDE solvers and simulation sensitivity analysis while
      avoiding the need to manually derive Jacobians.

   .. grid-item-card:: ML-surrogates
      :class-card: sd-border-0 torax-pillar-card
      :shadow: None

      ML-surrogate coupling for fast and accurate simulation is greatly facilitated
      by JAX's inherent support for neural network development and inference.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: torax-nav-grid

   .. grid-item-card:: :material-regular:`rocket_launch;1.5em` Installation & Quickstart
      :link: installation
      :link-type: ref
      :class-card: torax-nav-card

      Install TORAX from PyPI or source and run your first simulation.

   .. grid-item-card:: :material-regular:`school;1.5em` Tutorials & Examples
      :link: tutorials
      :link-type: ref
      :class-card: torax-nav-card

      Interactive Google Colab and Jupyter exercises for scenario design.

   .. grid-item-card:: :material-regular:`tune;1.5em` Simulation Configuration
      :link: configuration
      :link-type: ref
      :class-card: torax-nav-card

      Configure plasma profiles, transport models, sources, and solvers.

   .. grid-item-card:: :material-regular:`functions;1.5em` Physics & Derivations
      :link: equations
      :link-type: ref
      :class-card: torax-nav-card

      Governing 1D transport PDEs, FVM discretization, and physics models.

   .. grid-item-card:: :material-regular:`laptop_chromebook;1.5em` Developer Guide
      :link: structure
      :link-type: ref
      :class-card: torax-nav-card

      Codebase architecture, custom model integration, and JAX utilities.

   .. grid-item-card:: :material-regular:`groups;1.5em` Community & Roadmap
      :link: contributing
      :link-type: ref
      :class-card: torax-nav-card

      Development roadmap, contribution guidelines, and how to cite TORAX.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   overview
   installation
   quickstart

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials & Examples

   tutorials

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   configuration
   running
   running_programmatically
   output
   plotting
   using_jax

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Physics & Derivations

   equation_summary
   solver_details
   physics_models
   glossary

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer Guide

   code_structure
   fvm
   model_integration
   jax_classes
   jitted_simulation
   explicit_geometry

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Community & Project

   roadmap
   contributing
   contribution_tips
   citing
   contact

.. _JAX: https://jax.readthedocs.io/
