TORAX: Tokamak transport simulation in JAX
==========================================

Test: does this show up?

TORAX is a differentiable tokamak core transport simulator aimed for fast and
accurate forward modelling, pulse-design, trajectory optimization, and controller
design workflows. TORAX is written in Python using the JAX_ library.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Flexible
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Python facilitates coupling within various workflows and to additional
      physics models. Easy to install and JAX can seamlessly execute
      on multiple backends including CPU and GPU.

   .. grid-item-card:: Fast and auto-differentiable
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX provides just-in-time compilation for fast runtimes. JAX auto-differentiability
      enables gradient-based nonlinear PDE solvers and simulation sensitivity analysis while
      avoiding the need to manually derive Jacobians.

   .. grid-item-card:: ML-surrogates
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      ML-surrogate coupling for fast and accurate simulation is greatly facilitated
      by JAX's inherent support for neural network development and inference.

.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting Started
      :columns: 12 6 6 4
      :link: beginner-guides
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` User Guides
      :columns: 12 6 6 4
      :link: user-guides
      :link-type: ref
      :class-card: user-guides

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Developer Docs
      :columns: 12 6 6 4
      :link: developer-guides
      :link-type: ref
      :class-card: developer-docs


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   overview
   installation
   quickstart
   faq

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Further Resources

   user_guides
   developer_guides
   tutorials
   roadmap
   contributing
   citing
   contact

.. toctree::
   :hidden:
   :maxdepth: 1

   glossary

.. _JAX: https://jax.readthedocs.io/
