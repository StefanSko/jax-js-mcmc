# Testing Suite and Motivation

This project uses a layered test suite to validate both the physics of HMC and
the user-facing API. The layers are intentional: earlier tests protect core
physics invariants, later tests protect statistical behavior and interfaces.

## Why a layered suite

HMC correctness depends on symplectic integration and a valid Metropolis
correction. If those fail, later statistical tests can give false confidence.
We therefore run physics tests first, then posterior tests, then reference
comparisons.

## Test tiers

### 1) Physics invariants (must pass first)

These are the foundation. They validate properties of the leapfrog integrator
and Metropolis step that HMC relies on.

What we check:
- Energy conservation (drift scales as O(epsilon^2), small step sizes conserve).
- Reversibility (forward then backward returns to start).
- Volume preservation (Jacobian determinant is ~1). This is a smoke check with
  relaxed tolerance because finite-difference Jacobians are noisy in float32.
- Detailed balance (acceptance behavior matches Metropolis rules).
- Analytic trajectories (harmonic oscillator, free fall).
- Kinetic energy math (including pytree inputs).

Why it matters:
If any of these fail, chain correctness is not guaranteed even if posterior
tests look good.

### 2) Posterior tests (known distributions)

These validate that HMC recovers known distributions (e.g., multivariate normal,
Neal's funnel, banana). They exercise adaptation and tree handling in realistic
settings.

Why it matters:
Physics-correct integrators can still be wired incorrectly at the sampler
level. Posterior tests catch those wiring and adaptation errors.

### 3) Diagnostics tests

Diagnostics (R-hat, ESS, summary statistics) are part of the public API. These
tests ensure they accept the same data structures produced by `hmc` and return
reasonable values on known inputs.

### 4) Reference comparisons (optional, slower)

These compare against external reference implementations (e.g., NumPyro or
BlackJAX) when available. They are helpful for deeper validation but are not
required for every change.

## Running tests

Install dependencies first:

```bash
pnpm install
```

Then:

```bash
# Physics tests (run first)
pnpm test tests/physics

# All tests
pnpm test

# Only known posterior tests
pnpm test tests/posteriors

# Reference comparisons (optional)
pnpm test tests/reference
```

## Practical notes

- Tests run in float32; tolerances are sized for WebGPU/WASM behavior.
- Finite-difference Jacobians are noisy; volume preservation is a smoke check,
  not a strict proof.
- Always call `init()` and set a device (tests default to CPU for determinism).
- Use `.ref` for jax-js arrays when reusing values to avoid move-semantic
  errors. See `docs/JAX-JS-MEMORY.md` for details.

## Adding new tests

When adding tests, keep them deterministic (seeded RNG), small enough to run in
CI, and aligned with the tiering above. Physics tests should fail before any
new implementation is written, per the TDD rule.
