# jax-js-mcmc Design Document

**Date:** 2026-01-16
**Status:** Draft
**Repository:** github.com/StefanSko/jax-js-mcmc

## Overview

A standalone MCMC sampling library for jax-js. Provides HMC sampling with automatic step size adaptation for any differentiable log probability function.

This library is independent of any modeling DSL - it just takes a `logProb` function and returns samples. Can be proposed upstream to the jax-js org once mature.

## Prerequisites

**Before implementing, read:**
- **[JAX-JS-MEMORY.md](JAX-JS-MEMORY.md)** - Critical memory management patterns for jax-js

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

---

## API Specification

### Types

```typescript
// logProb returns scalar Array (0-dim float32), NOT JS number
type LogProbFn = (params: JsTree<Array>) => Array;

// JsTree = nested object/array of Array leaves
type JsTree<T> = T | { [key: string]: JsTree<T> } | JsTree<T>[];

interface HMCOptions {
  // Required (no defaults)
  initialParams: JsTree<Array>;
  key: Array;  // PRNGKey
  numSamples: number;

  // Optional with defaults
  numWarmup?: number;           // Default: 1000
  numLeapfrogSteps?: number;    // Default: 25
  numChains?: number;           // Default: 1
  initialStepSize?: number;     // Default: 0.1 (overridden by init heuristic)
  targetAcceptRate?: number;    // Default: 0.8
  adaptMassMatrix?: boolean;    // Default: true
}

interface HMCResult {
  draws: JsTree<Array>;         // Shape: [numChains, numSamples, ...paramShape]
  stats: {
    acceptRate: Array;          // Per-chain + mean
    stepSize: Array;            // Per-chain final step sizes
    massMatrix?: JsTree<Array>; // Per-chain diagonal mass matrices
  };
}
```

### Basic Usage

```typescript
import { hmc } from "jax-js-mcmc";
import { numpy as np, random, grad } from "@jax-js/jax";

// Any log probability function (must return scalar Array)
const logProb = (params: { mu: Array; sigma: Array }) => {
  const priorMu = params.mu.ref.mul(params.mu).mul(-0.5).div(25).sum();  // N(0,5)
  const priorSigma = params.sigma.mul(-1).sum();  // Exp(1)
  return priorMu.add(priorSigma);
};

const result = await hmc(logProb, {
  numSamples: 1000,
  numWarmup: 500,
  numLeapfrogSteps: 25,
  initialParams: { mu: np.zeros([1]), sigma: np.ones([1]) },
  key: random.key(42),
});

// Access results
result.draws;        // { mu: Array, sigma: Array } shape [1, 1000, 1]
result.stats;        // { acceptRate, stepSize, ... }
```

### Multiple Chains

```typescript
const result = await hmc(logProb, {
  numSamples: 1000,
  numChains: 4,
  key: random.key(42),
  initialParams: { x: np.zeros([2]) },
});

// result.draws.x shape: [4, 1000, 2]  (numChains, numSamples, paramShape)
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
//   mu:    { mean, sd, median, q5, q25, q75, q95, rhat, ess },
//   sigma: { mean, sd, median, q5, q25, q75, q95, rhat, ess },
// }
```

---

## Algorithm Details

### Step-Size Adaptation (Nesterov Dual Averaging)

Per-chain adaptation during warmup, freeze at end.

```
Hyperparameters:
  gamma = 0.05
  t0 = 10
  kappa = 0.75
  mu = log(10 * initialStepSize)  // center

State:
  logStepSize: number
  logStepSizeAvg: number
  hSum: number
  iteration: number

Update (each warmup step):
  hSum += targetAcceptRate - acceptProb
  logStepSize = mu - (sqrt(iteration) / gamma) * hSum
  logStepSizeAvg += iteration^(-kappa) * (logStepSize - logStepSizeAvg)

Final:
  stepSize = exp(logStepSizeAvg)  // use averaged value
```

### Step-Size Initialization Heuristic

```
1. Start with initialStepSize
2. Run 1 leapfrog step, compute acceptProb
3. While acceptProb > 0.8: double stepSize
   While acceptProb < 0.2: halve stepSize
4. Clamp to [1e-4, 1.0]
```

### Mass Matrix Adaptation (Diagonal, Welford)

Per-chain, during warmup only.

```
State:
  count: number
  mean: JsTree<Array>
  m2: JsTree<Array>

Update (each warmup sample):
  count += 1
  delta = sample - mean
  mean += delta / count
  delta2 = sample - mean
  m2 += delta * delta2

Final:
  variance = m2 / (count - 1)
  massMatrix = variance + 1e-5  // jitter for stability
```

**Usage in HMC:**
- Momentum sampling: `p ~ N(0, diag(massMatrix))`
- Kinetic energy: `K(p) = 0.5 * sum(p^2 / massMatrix)`

### Warmup Schedule (v1 Simple)

- Adapt step size + mass matrix at every warmup iteration
- Freeze both at end of warmup
- No windows (Stan-style windows deferred to v2)

### Multi-Chain Behavior

```
1. Split base key into numChains keys
2. Run chains independently (can parallelize later)
3. Stack results: draws shape [numChains, numSamples, ...paramShape]
4. Stats include per-chain arrays + means
```

---

## Diagnostics Specification

### R-hat (Split-Rhat, Gelman-Rubin)

- Split each chain in half → 2*numChains half-chains
- Compute between-chain variance B and within-chain variance W
- R-hat = sqrt((var_hat+ / W)) where var_hat+ = (n-1)/n * W + B/n
- No rank-normalization in v1

### ESS (Autocorrelation-Based)

- Use Geyer's initial positive sequence
- Report bulk ESS only (no tail ESS in v1)
- ESS = n * numChains / (1 + 2 * sum(autocorrelations))

### Summary

```typescript
interface ParamSummary {
  mean: number;
  sd: number;
  median: number;
  q5: number;
  q25: number;
  q75: number;
  q95: number;
  rhat: number;
  ess: number;
}
```

---

## Test Specifications

### Physics Test Tolerances (float32-friendly)

| Test | Tolerance |
|------|-----------|
| Reversibility | 1e-5 |
| Volume preservation (det J) | 1e-4 |
| Energy drift scaling ratio | 0.25 ± 0.2 |
| Detailed balance bins | ±0.1 accept rate |

### Posterior Test Configuration

| Posterior | numSamples | numWarmup | numChains |
|-----------|------------|-----------|-----------|
| MVN | 2000 | 1000 | 4 |
| Neal's funnel | 2000 | 1500 | 4 |
| Banana | 2000 | 1000 | 4 |

### Posterior Test Tolerances

| Test | Tolerance |
|------|-----------|
| MVN mean | 5% relative |
| MVN covariance | 10% relative |
| Neal's funnel mean | 0.25 absolute |
| Neal's funnel std | 0.35 absolute |

---

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
│   │   ├── volume-preservation.test.ts
│   │   └── detailed-balance.test.ts
│   ├── posteriors/      # Known posterior tests
│   │   ├── multivariate-normal.test.ts
│   │   ├── neals-funnel.test.ts
│   │   └── banana.test.ts
│   ├── diagnostics.test.ts
│   └── reference/       # Blue/green against NumPyro/BlackJAX
│       └── reference-comparison.test.ts
├── docs/
│   ├── DESIGN.md        # This file
│   └── JAX-JS-MEMORY.md # Memory management guide (READ FIRST)
├── CLAUDE.md            # Agent instructions
├── package.json
└── tsconfig.json
```

### Leapfrog Integrator (~100 lines)

```typescript
function leapfrog(
  position: JsTree<Array>,
  momentum: JsTree<Array>,
  gradLogProb: (p: JsTree<Array>) => JsTree<Array>,
  stepSize: number,
  numSteps: number,
  massMatrix?: JsTree<Array>,
): [JsTree<Array>, JsTree<Array>] {
  // Half step momentum
  // Full steps position + momentum
  // Half step momentum
  // IMPORTANT: Use tree.ref() for arrays that need to survive operations
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

---

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

---

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

---

## Dependencies

```json
{
  "name": "jax-js-mcmc",
  "version": "0.1.0",
  "peerDependencies": {
    "@jax-js/jax": ">=0.1.0"
  },
  "devDependencies": {
    "vitest": "^2.0.0",
    "typescript": "^5.3.0"
  }
}
```

---

## Future (v2+)

- NUTS (No-U-Turn Sampler) - automatic trajectory length
- Full mass matrix (dense covariance estimation)
- Windowed adaptation (Stan-style warmup schedule)
- Propose upstream to jax-js org as `@jax-js/mcmc`
