import sphinx.ext.autodoc

.. _fvm:

The 1D finite volume element (fvm) library
##########################################

The `torax.fvm` library can be thought of as a distinct submodule of
TORAX that is agnostic to the Tokamak modeling problem. The `fvm`
module abstracts the problem to a coupled set of one dimensional
convection-diffusion PDEs with diffusion, convection, implicit and
explicit sources, and transient terms. In principle, this module could
be used to solve other differential equations of this form---all of
the calculations of the values of the coefficients to make them describe
the tokamak core transport equations are done in other
parts of TORAX.

The `fvm` module can be thought of as similar in purpose to
`fipy`. Both solve differential equations with the same structure
(diffusion, convection etc terms describing fluid dynamics). See
:ref:`solver_details` for further mathematical details.
It is instructive to also compare the differences between `fipy` and `fvm`.
`fipy` is designed to be much more general and to be applied to a
wider variety of PDEs and with support for higher dimensions and
arbitrary mesh topologies. `torax.fvm` is much less general ---it
avoids hard-coded Tokamak physics into `torax.fvm` but has not attempted
to include any more features than needed for the Tokamak core transport
problem. `torax.fvm` thus supports only one topology, the 1-D
grid topology. However, within this restricted domain, `torax.fvm`
offers a variety of solver techniques not available in `fipy`,
especially solver techniques that are possible only with differentiable
solvers, in our case enabled by JAX. For example, in `torax.fvm` the
diffusion etc. coefficients are not raw numerical values, they are JAX
expressions, so when solvers like Newton-Raphson take derivatives to form
a Jacobian, they can actually differentiate through the diffusion
coefficients to form derivatives that are functions of more fundamental
underlying physical properties of the Tokamak.

The fvm module can roughly be categorized into three kinds of
submodules:

(1) Modules that provide solver functionality:

    - `implicit_solve_block` : Linear solver
    - `newton_raphson_solve_block` : Nonlinear Newton-Raphson solver
    - `optimizer_solve_block` : Nonlinear optimization-based solver

(2) Modules that define types. Users of the `fvm` module still need to import
    most of these to construct arguments and write type hints but they are not
    quite as front and center:

    - `block_1d_coeffs`
    - `cell_variable`
    - `enums`

(3) The remaining files are implementation files that do things like
    define the residual functions for the Newton method, perform discretization
    to construct theta method equations, etc.

Future developer documentation will describe in detail on how / why the exact
current implementation was derived.
