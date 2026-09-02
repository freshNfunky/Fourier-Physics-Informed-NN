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

## Precisation (Felix, refined)
- Bernoulli is the energy-conservation law, mathematically correct in itself, but
  partly taught and applied wrongly.
- The energy balance is the actual problem: in a closed system the pressure
  cannot be restored, because acceleration work in the total balance needs a
  gradient (Gefaelle), which is necessary to supply the inertial force as the
  reaction (actio/reactio). By the laws of dynamics, acceleration work can only
  exist when the total balance of static forces is NOT zero, yet most who teach
  Bernoulli / potential flow assume it is zero.
- Thesis (in the preprint): that cannot hold. A residual balance of kinetic
  energy must always remain as a residual turbulence, so an entropy increase
  necessarily occurs for the equation to balance. In other words, the idealized
  loss-free Bernoulli / potential-flow picture is inconsistent with the dynamics
  of a real closed cycle; dissipation (entropy production) is mandatory.
- Honest note for the resumed discussion: within its stated inviscid, loss-free
  assumptions Bernoulli is an exact first integral and self-consistent; the
  thesis is really about the physical *realizability* of those assumptions in a
  dynamic closed cycle. That is an arguable and interesting line (cf. d'Alembert's
  paradox: potential flow predicts zero drag), to be engaged against the preprint's
  specific mechanism.

## TODO when resumed
- Read the preprint in full and engage its specific argument.
- Optionally open this as a GitHub issue once the repo is pushed.
