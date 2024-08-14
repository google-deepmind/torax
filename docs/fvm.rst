import sphinx.ext.autodoc
.. _fvm:

The 1D finite volume element (fvm) library
##########################################

The `torax.fvm` library can be thought of as a distinct submodule of
torax that is agnostic to the Tokamak modeling problem. The `fvm`
module abstracts the problem to the level of fluid dynamics in
one dimension with diffusion, convection, implicit and explicit
source, and transient terms. In principle, this module could be
used to solve other differential equations of this form---all of
the calculations of the values of the coefficients to make them
describe Tokamak plasma physics equations are done in other
parts of `torax`.

The `fvm` module can be thought of as similar in purpose to
`fipy`. Both solve differential equations with the same structure
(diffusion, convection etc terms describing fluid dynamics).
It is instructive to also compare their differences.
`fipy` is designed to be much more general and to be applied to
any fluid flow problem. In particular, fluid flow allows arbitrary
mesh topologies. `torax.fvm` is much less generally---it avoids
hard-coded Tokamak physics into `torax.fvm` but has not attempted
to include any more features than needed for the Tokamak transport
problem. `torax.fvm` thus supports only one topology, the 1-D
grid topology. However, within this restricted domain, `torax.fvm`
offers a variety of solver techniques not available in `fipy`,
especially solver techniques that are posisble only with JAX.
For example, in `torax.fvm` the diffusion etc. coefficients are
not raw numerical values, they are JAX expressions, so when
solvers like Newton-Raphson take derivatives to form a Jacobian,
they can actually differentiate through the diffusion coefficients
to form derivatives that are functions of more fundamental
underlying physical properties of the Tokamak.

The fvm module can roughly be categorized into three kinds of
submodules:

(1) Highly user-facing modules that provide solver functionality:
.. comment:: api-doc makes a file for all of torax.fvm, not the
  individual submodules, so I can't make these list entries be
  links.

     - `implicit_solve_block` : Linear solver
     - `newton_raphson_solve_block` : Newton-Raphson solver
     - `optimizer_solve_block` : Optimization-based solver

(2) Moderately user-facing modules that define types. Users of the
  `fvm` module still need to import most of these to construct arguments
  and write type hints but they are not quite as front and center:

     - `block_1d_coeffs`
     - `cell_variable`
     - `enums`

(3) The remaining files are implementation files that do things like
define the residual functions for the Newton method, perform discretization
to construct theta method equations, etc.

Categories (1) and (2) are documented adequately from the API
reference but category (3) is the result of extensive mathematical
derivation and its behavior is strongly affected by several very
subtle points. This developer documentation describes in detail
how / why the exact current implementation was derived.

Under construction.

