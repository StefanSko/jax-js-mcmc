# Agent Instructions for jax-js-mcmc

## Project Overview

jax-js-mcmc is a standalone HMC sampling library for jax-js (JAX in the browser).
It provides MCMC inference for any differentiable log probability function.

## CRITICAL: Read Before Implementing

**Before writing any code, read these docs:**

1. **[docs/JAX-JS-MEMORY.md](docs/JAX-JS-MEMORY.md)** - jax-js uses move semantics. Every operation consumes its inputs. You MUST understand `.ref` patterns or your code will fail with "tracer freed" errors.

2. **[docs/DESIGN.md](docs/DESIGN.md)** - Full API specification, algorithm details, and test tolerances.

## Reference: jax-js

This library builds on jax-js. Clone it for reference:

```bash
git clone https://github.com/ekzhang/jax-js.git /tmp/jax-js
```

Refer to /tmp/jax-js/src for:
- Array API conventions
- How grad/jit/vmap work
- Random number generation patterns

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

```bash
# Run physics tests only (must pass before anything else)
npm test tests/physics

# Run all tests
npm test

# Run in browser (WebGPU)
npm test:browser
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
- **Use `.ref` when arrays need to survive operations** (see JAX-JS-MEMORY.md)

## Key Files

- `src/leapfrog.ts` - Leapfrog integrator (core building block)
- `src/hmc.ts` - HMC sampler
- `src/adaptation.ts` - Step size and mass matrix adaptation
- `src/diagnostics.ts` - R-hat, ESS, summary
- `tests/physics/` - Physics invariant tests (TDD foundation)
- `tests/posteriors/` - Known analytical posteriors
- `tests/reference/` - Blue/green tests against NumPyro
- `docs/JAX-JS-MEMORY.md` - **Read this first** - memory management guide
- `docs/DESIGN.md` - Full specification
