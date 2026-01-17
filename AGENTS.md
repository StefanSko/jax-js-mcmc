# Agent Instructions for jax-js-mcmc

## Project Overview

jax-js-mcmc is a standalone HMC sampling library for jax-js (JAX in the browser).
It provides MCMC inference for any differentiable log probability function.

## Reference: jax-js

This library builds on jax-js. Clone it for reference:

```bash
git clone https://github.com/ekzhang/jax-js.git /tmp/jax-js
```

Refer to /tmp/jax-js/src for:
- Array API conventions
- How grad/jit/vmap work
- Random number generation patterns

See `docs/JAX-JS-MEMORY.md` for jax-js move semantics and `.ref` usage patterns.

## Types and Conventions

- `JsTree<Array>` means any nested object/array structure whose leaves are jax-js `Array` values.
- `logProb(params)` must return a scalar (0-dim) `Array` in float32 (not a JS number).
- `initialParams`, `key`, and `numSamples` are required inputs to `hmc`.
- Tests assume float32 behavior (WebGPU/WASM); tolerances reflect this.

## Adaptation Defaults (v1)

- Dual averaging per chain with `gamma = 0.05`, `t0 = 10`, `kappa = 0.75`, `mu = log(10 * initialStepSize)`.
- Step size init heuristic: start `initialStepSize`, double if acceptProb > 0.8, halve if < 0.2, clamp to `[1e-4, 1]`.
- Diagonal mass matrix with Welford variance during warmup only, jitter `+ 1e-5`, freeze after warmup.

## MANDATORY: Physics-Based TDD

This project follows strict physics-based TDD. You MUST NOT skip these steps.

### The Rule

**Write physics tests FIRST. Implementation code comes AFTER tests exist and fail.**

HMC correctness depends on Hamiltonian mechanics invariants:
1. Energy conservation (leapfrog)
2. Reversibility (leapfrog)
3. Volume preservation (symplectic property)
4. Detailed balance (Metropolis correction)

### Development Order (NON-NEGOTIABLE)

1. **Write physics tests** → they fail (no implementation yet)
2. **Implement leapfrog** → physics tests pass
3. **Write posterior tests** → they fail
4. **Implement HMC** → posterior tests pass
5. **Write reference tests** → they fail
6. **Tune implementation** → reference tests pass

### Test Commands

Before running tests, install dependencies with `pnpm install`.

```bash
# Run physics tests only (must pass before anything else)
pnpm test tests/physics

# Run all tests
pnpm test

# Run in browser (WebGPU)
pnpm test:browser
```

### CI Enforcement

CI runs tests in phases. Later phases only run if earlier phases pass:

1. Physics tests
2. Known posterior tests
3. Reference comparison tests

A PR cannot merge if physics tests fail.

## Code Style

- TypeScript with strict mode
- Pure functions where possible
- Match jax-js API conventions (e.g., `Array.add()` not `+`)
- Use jax-js tree utilities for nested parameter structures

## Key Files

- `src/leapfrog.ts` - Leapfrog integrator (core building block)
- `src/hmc.ts` - HMC sampler
- `src/adaptation.ts` - Step size and mass matrix adaptation
- `src/diagnostics.ts` - R-hat, ESS, summary
- `tests/physics/` - Physics invariant tests (TDD foundation)
- `tests/posteriors/` - Known analytical posteriors
- `tests/reference/` - Blue/green tests against NumPyro
