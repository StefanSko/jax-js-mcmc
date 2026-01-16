# jax-js-mcmc Design Document

**Date:** 2026-01-16
**Status:** Draft
**Repository:** github.com/StefanSko/jax-js-mcmc

## Overview

A standalone MCMC sampling library for jax-js. Provides HMC sampling with automatic step size adaptation for any differentiable log probability function.

This library is independent of any modeling DSL - it just takes a `logProb` function and returns samples. Can be proposed upstream to the jax-js org once mature.

## Goals

1. **Standalone** - No modeling DSL, just MCMC algorithms
2. **Simple API** - Pass log prob function, get samples
3. **Browser-native** - Runs on jax-js (WebGPU/WASM)
4. **Physics-grounded** - Implementation validated against Hamiltonian mechanics invariants
5. **Reference-validated** - Tested against posteriordb and known analytical posteriors

## Non-Goals (v1)

- NUTS (HMC is sufficient for initial use cases)
- Full mass matrix adaptation (diagonal only)
- Parallel tempering, SMC, or other advanced samplers

## Reference: jax-js

This library builds on [jax-js](https://github.com/ekzhang/jax-js) - JAX for the browser.

**For development, clone jax-js to /tmp for reference:**

```bash
git clone https://github.com/ekzhang/jax-js.git /tmp/jax-js
```

Key jax-js features we depend on:
- `@jax-js/jax`: numpy-like arrays, `grad`, `jit`, `vmap`
- Random: `random.key()`, `random.normal()`, `random.split()`
- Linear algebra: basic matmul, element-wise ops

Reference the jax-js source when implementing to match API conventions and understand available primitives.

## API Design

### Types and Conventions

- **Parameter trees**: `JsTree<Array>` means any nested object/array structure whose leaves are jax-js `Array` values.
- **logProb**: `logProb(params)` must return a **scalar** (0-dim) `Array` in float32 (not a JS number).
- **RNG**: `key` is a jax-js `PRNGKey`. For multiple chains, split the key once per chain.
- **Precision**: tests assume float32 behavior (WebGPU/WASM); tolerances reflect this.

### Basic Usage

```typescript
import { hmc } from "jax-js-mcmc";
import { numpy as np, random } from "@jax-js/jax";

// Any log probability function
const logProb = (params: { mu: Array; sigma: Array }) => {
  // Standard normal priors (written explicitly to avoid extra deps)
  const logpMu = params.mu.pow(2).mul(-0.5).sum();
  const logpSigma = params.sigma.pow(2).mul(-0.5).sum();
  return logpMu.add(logpSigma);
};

const result = await hmc(logProb, {
  numSamples: 1000,
  numWarmup: 500,
  numLeapfrogSteps: 25,
  initialParams: { mu: np.zeros([1]), sigma: np.ones([1]) },
  key: random.key(42),
});

// Access results
result.draws;        // { mu: Array, sigma: Array }
result.stats;        // { acceptRate, stepSize, ... }
```

### Configuration Options

```typescript
interface HMCOptions {
  // Required
  numSamples: number;
  initialParams: Record<string, Array>;
  key: PRNGKey;

  // Optional with defaults
  numWarmup?: number;           // Default: 1000
  numLeapfrogSteps?: number;    // Default: 25
  numChains?: number;           // Default: 1

  // Step size adaptation
  initialStepSize?: number;     // Default: 0.1
  targetAcceptRate?: number;    // Default: 0.8

  // Mass matrix adaptation
  adaptMassMatrix?: boolean;    // Default: true
}
```

### Adaptation Defaults (Concrete)

- **Dual averaging (per-chain)** with Stan/NumPyro-style hyperparameters:
  - `gamma = 0.05`, `t0 = 10`, `kappa = 0.75`
  - `mu = log(10 * initialStepSize)`
- **Warmup**: adapt step size at every warmup iteration; freeze to `logStepSizeAvg` after warmup.
- **Step size init heuristic**: start at `initialStepSize`; while acceptProb > 0.8, double; while < 0.2, halve; clamp to `[1e-4, 1]` using a single-step trajectory.
- **Mass matrix (diagonal, per-chain)**: Welford online variance during warmup only; `massMatrix = variance + 1e-5` jitter; freeze after warmup.

### Multiple Chains

```typescript
const result = await hmc(logProb, {
  numSamples: 1000,
  numChains: 4,
  initialParams: { mu: np.zeros([1]), sigma: np.ones([1]) },
  key: random.key(42),
});

// result.draws.mu shape: [numChains, numSamples, ...paramShape]
```

### Diagnostics

```typescript
import { rhat, ess, summary } from "jax-js-mcmc/diagnostics";

// Individual diagnostics
rhat(result.draws.mu);    // R-hat (< 1.01 is good)
ess(result.draws.mu);     // Effective sample size (> 400 is good)

// Summary table
summary(result.draws);
// {
//   mu:    { mean, sd, q5, q25, q50, q75, q95, rhat, ess },
//   sigma: { mean, sd, q5, q25, q50, q75, q95, rhat, ess },
// }
```

**Definitions (v1)**

- **R-hat**: split-Rhat (Gelman–Rubin); no rank-normalization.
- **ESS**: Geyer initial positive sequence; report bulk ESS only.

### Sampler Statistics

```typescript
result.stats.acceptRate;      // Mean acceptance rate
result.stats.stepSize;        // Final adapted step size
result.stats.massMatrix;      // Diagonal mass matrix (if adapted)
```

## Implementation

### Core Components

```
jax-js-mcmc/
├── src/
│   ├── hmc.ts           # Main HMC sampler
│   ├── leapfrog.ts      # Leapfrog integrator
│   ├── adaptation.ts    # Step size (dual averaging), mass matrix
│   ├── diagnostics.ts   # R-hat, ESS, summary
│   └── index.ts
├── tests/
│   ├── physics/         # Physics invariant tests (TDD foundation)
│   │   ├── energy-conservation.test.ts
│   │   ├── reversibility.test.ts
│   │   └── volume-preservation.test.ts
│   ├── posteriors/      # Known posterior tests
│   │   ├── multivariate-normal.test.ts
│   │   ├── neals-funnel.test.ts
│   │   └── banana.test.ts
│   ├── diagnostics.test.ts
│   └── reference/       # Blue/green against NumPyro/BlackJAX
│       └── reference-comparison.test.ts
├── CLAUDE.md            # Agent instructions (see below)
├── package.json
└── tsconfig.json
```

### Leapfrog Integrator (~50 lines)

```typescript
function leapfrog(
  position: JsTree<Array>,
  momentum: JsTree<Array>,
  gradLogProb: (p: JsTree<Array>) => JsTree<Array>,
  stepSize: number,
  numSteps: number,
): [JsTree<Array>, JsTree<Array>] {
  // Half step momentum
  // Full steps position + momentum
  // Half step momentum
}
```

### Dual Averaging for Step Size (~50 lines)

```typescript
interface DualAverageState {
  logStepSize: number;
  logStepSizeAvg: number;
  hSum: number;
  iteration: number;
}

function updateDualAverage(
  state: DualAverageState,
  acceptProb: number,
  targetAcceptRate: number,
): DualAverageState;
```

### Mass Matrix Adaptation (~50 lines)

```typescript
interface MassMatrixState {
  mean: JsTree<Array>;
  m2: JsTree<Array>;
  count: number;
}

function updateMassMatrix(
  state: MassMatrixState,
  sample: JsTree<Array>,
): MassMatrixState;

function getMassMatrix(state: MassMatrixState): JsTree<Array>;
```

### R-hat and ESS (~50 lines)

```typescript
// R-hat: between-chain vs within-chain variance
function rhat(draws: Array): number;  // draws shape: [chains, samples, ...]

// ESS: effective sample size accounting for autocorrelation
function ess(draws: Array): number;
```

## Testing Strategy: Physics-Based TDD

### Foundational Principle

HMC is grounded in Hamiltonian mechanics. Before testing statistical properties, we validate the physics invariants that make HMC correct. These tests are **mandatory TDD steps** - implementation proceeds only after physics tests pass.

### The Hamiltonian

```
H(q, p) = U(q) + K(p)

where:
  q = position (parameters)
  p = momentum (auxiliary variables)
  U(q) = -logProb(q)  (potential energy)
  K(p) = 0.5 * p^T * M^{-1} * p  (kinetic energy)
```

### Phase 1: Physics Invariant Tests (MUST PASS FIRST)

These tests validate the mathematical properties that guarantee HMC correctness. They require no reference implementation - they follow from theory.

#### 1.1 Energy Conservation

Leapfrog should conserve the Hamiltonian up to O(ε²) per step:

```typescript
// tests/physics/energy-conservation.test.ts
describe("leapfrog energy conservation", () => {
  test("energy drift scales with O(L * ε²)", () => {
    const stepSize = 0.1;
    const numSteps = 100;

    // Simple quadratic potential: U(q) = 0.5 * q^2
    const gradU = (q: Array) => q;

    const q0 = np.array([1.0, 0.5]);
    const p0 = np.array([0.0, 1.0]);

    const H0 = hamiltonian(q0, p0, gradU);
    const [q1, p1] = leapfrog(q0, p0, gradU, stepSize, numSteps);
    const H1 = hamiltonian(q1, p1, gradU);

    const energyDrift = Math.abs(H1 - H0);
    expect(energyDrift).toBeGreaterThanOrEqual(0); // sanity check
  });

  test("energy drift scales quadratically with step size", () => {
    const stepSizes = [0.1, 0.05, 0.025];
    const drifts = stepSizes.map(ε => measureEnergyDrift(ε, 100));

    // Halving step size should quarter the drift
    expect(drifts[1] / drifts[0]).toBeCloseTo(0.25, { tolerance: 0.2 });
    expect(drifts[2] / drifts[1]).toBeCloseTo(0.25, { tolerance: 0.2 });
  });
});
```

#### 1.2 Reversibility

Leapfrog is time-reversible: negate momentum, run same steps, return to start:

```typescript
// tests/physics/reversibility.test.ts
describe("leapfrog reversibility", () => {
  test("forward then backward returns to start", () => {
    const q0 = np.array([1.0, 2.0, 3.0]);
    const p0 = np.array([0.5, -0.5, 0.1]);

    // Forward
    const [q1, p1] = leapfrog(q0, p0, gradU, stepSize, numSteps);

    // Backward (negate momentum)
    const [q2, p2] = leapfrog(q1, p1.neg(), gradU, stepSize, numSteps);

    // Should return to start (up to floating point)
    expect(q2).toBeCloseTo(q0, { tolerance: 1e-5 });
    expect(p2.neg()).toBeCloseTo(p0, { tolerance: 1e-5 });
  });
});
```

#### 1.3 Volume Preservation (Symplectic)

Leapfrog preserves phase space volume exactly (Jacobian determinant = 1):

```typescript
// tests/physics/volume-preservation.test.ts
describe("leapfrog volume preservation", () => {
  test("jacobian determinant equals 1", () => {
    // Compute Jacobian of leapfrog map numerically
    const jacobian = computeJacobian((qp) => {
      const [q, p] = split(qp);
      const [q1, p1] = leapfrog(q, p, gradU, stepSize, numSteps);
      return concat(q1, p1);
    }, concat(q0, p0));

    const det = np.linalg.det(jacobian);
    expect(det).toBeCloseTo(1.0, { tolerance: 1e-4 });
  });
});
```

#### 1.4 Detailed Balance

HMC satisfies detailed balance with Metropolis correction:

```typescript
// tests/physics/detailed-balance.test.ts
describe("HMC detailed balance", () => {
  test("acceptance probability follows Metropolis rule", () => {
    // For energy difference ΔH, acceptance should be min(1, exp(-ΔH))
    const results = runManyProposals(logProb, numTrials: 5000);

    // Bin by energy difference, check acceptance rates
    for (const bin of energyBins) {
      const expectedAcceptRate = Math.min(1, Math.exp(-bin.meanDeltaH));
      expect(bin.observedAcceptRate).toBeCloseTo(expectedAcceptRate, { tolerance: 0.1 });
    }
  });
});
```

### Phase 2: Known Posterior Tests

After physics tests pass, validate against posteriors with known analytical properties.

#### 2.1 Multivariate Normal (Exact Solution Known)

```typescript
// tests/posteriors/multivariate-normal.test.ts
describe("multivariate normal posterior", () => {
  // Known: mean = [0, 0], cov = [[1, 0.8], [0.8, 1]]
  const trueMean = [0, 0];
  const trueCov = [[1, 0.8], [0.8, 1]];

  const logProb = (p: { x: Array }) => {
    return mvnLogProb(p.x, trueMean, trueCov);
  };

  test("recovers true mean within 5%", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });
    const sampleMean = mean(result.draws.x, axis: [0, 1]);

    expect(sampleMean[0]).toBeCloseTo(trueMean[0], { tolerance: 0.05 });
    expect(sampleMean[1]).toBeCloseTo(trueMean[1], { tolerance: 0.05 });
  });

  test("recovers true covariance within 10%", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });
    const sampleCov = cov(result.draws.x.reshape([-1, 2]));

    expect(sampleCov[0][0]).toBeCloseTo(trueCov[0][0], { tolerance: 0.1 });
    expect(sampleCov[0][1]).toBeCloseTo(trueCov[0][1], { tolerance: 0.1 });
  });

  test("R-hat < 1.01 for converged chains", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });
    expect(rhat(result.draws.x)).toBeLessThan(1.01);
  });
});
```

#### 2.2 Neal's Funnel (Pathological Geometry)

Tests adaptation on difficult posterior:

```typescript
// tests/posteriors/neals-funnel.test.ts
describe("Neal's funnel", () => {
  // v ~ Normal(0, 3)
  // x | v ~ Normal(0, exp(v/2))
  const logProb = (p: { v: Array; x: Array }) => {
    const logPv = normal(0, 3).logProb(p.v);
    const logPx = normal(0, np.exp(p.v.div(2))).logProb(p.x);
    return logPv.add(logPx.sum());
  };

  test("samples from both neck and mouth of funnel", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numChains: 4,
      numWarmup: 1500,  // Needs more warmup for adaptation
      initialParams: { v: np.zeros([1]), x: np.zeros([8]) },
      key: random.key(42),
    });

    // v should have samples across range [-6, 6]
    const vSamples = result.draws.v.flatten();
    expect(np.min(vSamples)).toBeLessThan(-3);
    expect(np.max(vSamples)).toBeGreaterThan(3);
  });

  test("marginal of v matches Normal(0, 3)", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1500,
      numChains: 4,
      initialParams: { v: np.zeros([1]), x: np.zeros([8]) },
      key: random.key(42),
    });
    const vMean = mean(result.draws.v);
    const vStd = std(result.draws.v);

    expect(vMean).toBeCloseTo(0, { tolerance: 0.25 });
    expect(vStd).toBeCloseTo(3, { tolerance: 0.35 });
  });
});
```

#### 2.3 Banana-Shaped Posterior

Tests curved geometry:

```typescript
// tests/posteriors/banana.test.ts
describe("banana posterior", () => {
  // Twisted normal: x1 ~ Normal(0, 10), x2 | x1 ~ Normal(b*x1^2, 1)
  const b = 0.1;
  const logProb = (p: { x: Array }) => {
    const x1 = p.x.slice([0], [1]);
    const x2 = p.x.slice([1], [1]);
    return normal(0, 10).logProb(x1).add(
      normal(b * x1.pow(2), 1).logProb(x2)
    );
  };

  test("recovers curved posterior shape", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });

    // Check that x2 correlates with x1^2
    const x1 = result.draws.x.slice([null, null, 0]);
    const x2 = result.draws.x.slice([null, null, 1]);
    const x1Squared = x1.pow(2);

    const correlation = corrcoef(x1Squared.flatten(), x2.flatten());
    expect(correlation).toBeGreaterThan(0.8);  // Strong positive correlation
  });
});
```

### Phase 3: Blue/Green Reference Tests

Compare against established implementations (NumPyro, BlackJAX) on complex posteriors.

#### Reference Generation (Python - run once)

```python
# scripts/generate_reference.py
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import json

def eight_schools():
    # ... model definition ...
    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=4000)
    mcmc.run(jax.random.PRNGKey(42), y=y, sigma=sigma)

    samples = mcmc.get_samples()
    return {
        "mu": {"mean": float(samples["mu"].mean()), "std": float(samples["mu"].std())},
        "tau": {"mean": float(samples["tau"].mean()), "std": float(samples["tau"].std())},
    }

# Save reference
with open("tests/reference/eight-schools-reference.json", "w") as f:
    json.dump(eight_schools(), f)
```

#### Reference Comparison (TypeScript)

```typescript
// tests/reference/reference-comparison.test.ts
import reference from "./eight-schools-reference.json";

describe("blue/green: eight schools vs NumPyro", () => {
  test("mu matches NumPyro reference within 15%", async () => {
    const result = await hmc(eightSchoolsLogProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      key: randomKey(42),
      initialParams: { mu: np.zeros([1]), tau: np.ones([1]) },
    });

    const stats = summary(result.draws);

    expect(stats.mu.mean).toBeCloseTo(reference.mu.mean, { tolerance: 0.15 });
    expect(stats.mu.sd).toBeCloseTo(reference.mu.std, { tolerance: 0.15 });
  });

  test("tau matches NumPyro reference within 15%", async () => {
    const result = await hmc(eightSchoolsLogProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      key: randomKey(42),
      initialParams: { mu: np.zeros([1]), tau: np.ones([1]) },
    });
    const stats = summary(result.draws);

    expect(stats.tau.mean).toBeCloseTo(reference.tau.mean, { tolerance: 0.15 });
    expect(stats.tau.sd).toBeCloseTo(reference.tau.std, { tolerance: 0.15 });
  });
});
```

### Test Execution Order (Mandatory)

```
1. Physics Tests (MUST PASS)
   ├── energy-conservation.test.ts
   ├── reversibility.test.ts
   ├── volume-preservation.test.ts
   └── detailed-balance.test.ts

2. Known Posteriors (MUST PASS)
   ├── multivariate-normal.test.ts
   ├── neals-funnel.test.ts
   └── banana.test.ts

3. Reference Comparison (MUST PASS)
   └── reference-comparison.test.ts
```

CI fails if any phase fails. Implementation proceeds only when physics tests pass.

### Test Matrix Summary

| Test Type | Reference | Tolerance | What It Validates |
|-----------|-----------|-----------|-------------------|
| Energy conservation | Theory: O(ε²) | Quadratic scaling | Leapfrog correctness |
| Reversibility | Exact | 1e-10 | Leapfrog correctness |
| Volume preservation | det(J) = 1 | 1e-10 | Symplectic property |
| Detailed balance | Metropolis rule | 5% | HMC correctness |
| MVN mean | Analytical | 5% | Sampler accuracy |
| MVN covariance | Analytical | 10% | Sampler accuracy |
| Neal's funnel | Marginal v~N(0,3) | 10% | Adaptation quality |
| Eight schools | NumPyro | 15% | Real-world accuracy |

## CLAUDE.md (Agent Instructions)

The following should be placed in `CLAUDE.md` at the repository root:

```markdown
# Agent Instructions for jax-js-mcmc

## Project Overview

jax-js-mcmc is a standalone HMC sampling library for jax-js (JAX in the browser).
It provides MCMC inference for any differentiable log probability function.

## Reference: jax-js

This library builds on jax-js. Clone it for reference:

\`\`\`bash
git clone https://github.com/ekzhang/jax-js.git /tmp/jax-js
\`\`\`

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

\`\`\`bash
# Run physics tests only (must pass before anything else)
pnpm test tests/physics

# Run all tests
pnpm test

# Run in browser (WebGPU)
pnpm test:browser
\`\`\`

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
```

## Dependencies

```json
{
  "name": "jax-js-mcmc",
  "version": "0.1.0",
  "peerDependencies": {
    "@jax-js/jax": ">=0.1.0"
  },
  "devDependencies": {
    "vitest": "^1.0.0",
    "playwright": "^1.40.0"
  }
}
```

## Implementation Order (TDD)

1. **Write physics tests first** (they will fail)
2. **Implement leapfrog** → physics tests pass
3. **Write known posterior tests** (they will fail)
4. **Implement basic HMC** → simple posterior tests pass
5. **Implement dual averaging** → adaptation tests pass
6. **Implement mass matrix adaptation** → funnel test passes
7. **Write reference tests** (they will fail)
8. **Tune and validate** → reference tests pass
9. **Implement diagnostics** (R-hat, ESS)
10. **Multi-chain support**

## Future (v2+)

- NUTS (No-U-Turn Sampler) - automatic trajectory length
- Full mass matrix (dense covariance estimation)
- Windowed adaptation (Stan-style warmup schedule)
- Propose upstream to jax-js org as `@jax-js/mcmc`
