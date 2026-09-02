# Parked note: critique of potential flow / Bernoulli energy interpretation

Status: parked for a dedicated discussion. Captured here so it is not lost, and
because it bears on the compressible-vs-incompressible framing of the PINN work.

## Felix's position (as stated)
- Potential flow, including the Bernoulli energy view, is misinterpreted in the
  mainstream fluid-mechanics school, and the paradigm persists stubbornly.
- Bernoulli conflates static pressure with a force field like gravity. But
  gravitational potential energy (rho*g*z) is real stored energy: height in a
  gravitational field converts to real energy. Static pressure does not have
  that property; it is a bookkeeping quantity, and the actual work arises
  elsewhere, not inside the fluid.
- Preprint: https://doi.org/10.5281/zenodo.21337311  (to be read in full when we
  resume this thread).

## Discussion summary (both sides, honest)
- Kernel of truth: the pressure term in Bernoulli is NOT intrinsic stored energy
  like (1/2) rho v^2 or rho g z. It is flow work / pressure work (integral of
  dp/rho), i.e. work exchanged with neighboring fluid across the element's
  boundary. Calling static pressure "energy stored in the fluid" is
  interpretationally sloppy, and "the work happens elsewhere" points exactly at
  this: the term accounts for work done by/on neighbors, not a reservoir inside
  the element. The gravity analogy is imperfect: rho g z is a genuine potential,
  p is a work term.
- Where the strong claim overreaches: this does not make Bernoulli false. It is a
  correct first integral of the Euler momentum equation along a streamline; the
  pressure term correctly books the pressure work, and it is validated within its
  assumptions (steady, inviscid, incompressible, along a streamline). The
  framework is an idealization with known limits, not a delusion.
- Strongest form of the critique: it lands on the incompressible/inviscid
  idealization and its careless energy interpretation. Incompressible static
  pressure is an instantaneous global constraint (Poisson equation, infinite
  sound speed) = "teleportation" by construction; a compressible, finite-speed
  formulation is the more honest physics.

## Relevance to the PINN track
Favor compressible / finite-propagation formulations so the model respects
"nothing teleports." This is a concrete framing hook, not just philosophy.

## TODO when resumed
- Read the preprint in full and engage its specific argument.
- Optionally open this as a GitHub issue once the repo is pushed.
