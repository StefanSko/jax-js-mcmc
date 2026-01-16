# jax-js-mcmc

Hamiltonian Monte Carlo (HMC) for [jax-js](https://github.com/ekzhang/jax-js): Bayesian inference in the browser and Node.

## Status

**Alpha** — core HMC + diagnostics are implemented, but the API may change. See `docs/DESIGN.md` for the full spec and testing strategy.

## Features (v1)

- HMC with dual-averaging step size adaptation
- Diagonal mass matrix adaptation
- Diagnostics: split-Rhat, bulk ESS, summary statistics
- Physics-based TDD validation (energy, reversibility, volume preservation)

## Installation

```bash
pnpm add jax-js-mcmc @jax-js/jax
```

## Quickstart (Node / TypeScript)

```typescript
import { hmc } from "jax-js-mcmc";
import { init, defaultDevice, numpy as np, random, type Array } from "@jax-js/jax";

await init();
defaultDevice("cpu");

const logProb = (params: { x: Array }) => {
  // Return a scalar Array (0-dim), not a JS number.
  return params.x.ref.mul(params.x).mul(-0.5).sum();
};

const result = await hmc(logProb, {
  numSamples: 1000,
  numWarmup: 500,
  numLeapfrogSteps: 25,
  initialParams: { x: np.zeros([2]) },
  key: random.key(42),
});

console.log(result.stats.acceptRate);
console.log(await result.draws.x.jsAsync());
```

## Quickstart (Browser)

You can run this with any bundler (Vite, Parcel, etc.) that resolves ESM dependencies.
See `examples/browser` for a complete Vite setup.

```html
<!doctype html>
<html>
  <body>
    <script type="module">
      import { hmc } from "jax-js-mcmc";
      import {
        init,
        defaultDevice,
        numpy as np,
        random,
      } from "@jax-js/jax";

      await init();
      // Use WebGPU if available, else CPU.
      try {
        defaultDevice("webgpu");
      } catch {
        defaultDevice("cpu");
      }

      const logProb = (params) =>
        params.x.ref.mul(params.x).mul(-0.5).sum();

      const result = await hmc(logProb, {
        numSamples: 500,
        numWarmup: 500,
        numLeapfrogSteps: 25,
        initialParams: { x: np.zeros([2]) },
        key: random.key(0),
      });

      console.log("accept rate", result.stats.acceptRate);
      console.log("draws", await result.draws.x.jsAsync());
    </script>
  </body>
</html>
```

## API Expectations (v1)

- `logProb(params)` returns a scalar `Array` in float32.
- `initialParams`, `key`, and `numSamples` are required.
- Parameter trees are nested objects/arrays whose leaves are `Array` values.

## Diagnostics

```typescript
import { rhat, ess, summary } from "jax-js-mcmc/diagnostics";

const rh = rhat(result.draws.x);
const e = ess(result.draws.x);
const stats = summary(result.draws);
```

## Quickstart (CommonJS)

If you’re in a CJS environment, use dynamic `import()`:

```javascript
const { init, defaultDevice, numpy: np, random } = await import("@jax-js/jax");
const { hmc } = await import("jax-js-mcmc");

await init();
defaultDevice("cpu");

const logProb = (params) => params.x.ref.mul(params.x).mul(-0.5).sum();
const result = await hmc(logProb, {
  numSamples: 1000,
  numWarmup: 500,
  initialParams: { x: np.zeros([2]) },
  key: random.key(42),
});
```

## Troubleshooting

- `init()` must be called before using jax-js; device selection happens after init.
- If WebGPU fails, fall back to `defaultDevice("cpu")`.
- WebGPU may be blocked by browser settings; try a recent Chrome/Edge and enable WebGPU.
- If you see `Key must have at least one dimension`, ensure you used `random.key(seed)` (a scalar seed).
- Very slow first run is normal due to shader compilation.

## Related

- [jax-js](https://github.com/ekzhang/jax-js) — JAX in the browser

## License

MIT
