# jax-js-mcmc

HMC sampling library for [jax-js](https://github.com/ekzhang/jax-js) - Bayesian inference in the browser.

## Status

**Work in progress** - See [docs/DESIGN.md](docs/DESIGN.md) for the full design document.

## Overview

A standalone MCMC sampling library for jax-js. Provides HMC sampling with automatic step size adaptation for any differentiable log probability function.

Requires `@jax-js/jax` as a peer dependency.

```typescript
import { hmc } from "jax-js-mcmc";
import { numpy as np, random } from "@jax-js/jax";

const logProb = (params) => {
  // Your log probability function
};

const result = await hmc(logProb, {
  numSamples: 1000,
  numWarmup: 500,
  initialParams: { x: np.zeros([10]) },
  key: random.key(42),
});
```

## Features

- HMC with dual averaging step size adaptation
- Diagonal mass matrix adaptation with windowed warmup
- Multi-chain sampling
- Diagnostics: R-hat, ESS, summary statistics
- Physics-based TDD validation

## Diagnostics

```typescript
import { ess, rhat, summary } from "jax-js-mcmc";

const stats = summary(result.draws);
const rhatValue = rhat(result.draws.x);
const essValue = ess(result.draws.x);
```

## Related

- [jax-js](https://github.com/ekzhang/jax-js) - JAX in the browser

## License

MIT
