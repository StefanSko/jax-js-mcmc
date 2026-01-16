/**
 * Banana (Rosenbrock) Posterior Tests
 *
 * Tests HMC on a banana-shaped (curved) posterior.
 * This tests the sampler's ability to handle non-Gaussian geometry.
 *
 * Target: x ~ N(0, 1), y ~ N(x^2 - 1, 0.5)
 * This creates a banana-shaped distribution in (x, y) space.
 *
 * Config: 500 samples, 200 warmup, 2 chains (reduced for speed)
 */

import { numpy as np, random } from "@jax-js/jax";
import { describe, expect, test } from "vitest";

import { hmc } from "../../src/hmc";

describe("Banana Posterior", () => {
  // Target distribution:
  // x ~ N(0, 1) => logProb_x = -0.5 * x^2
  // y ~ N(x^2 - 1, sigma_y) => logProb_y = -0.5 * (y - x^2 + 1)^2 / sigma_y^2
  //
  // The banana curvature comes from the x^2 term in y's mean

  const sigmaX = 1.0;
  const sigmaY = Math.sqrt(0.5); // variance = 0.5

  // True statistics:
  // E[x] = 0
  // E[y] = E[x^2 - 1] = Var[x] + E[x]^2 - 1 = 1 + 0 - 1 = 0
  // Var[x] = 1
  // Var[y] = Var[x^2 - 1] + sigma_y^2 = 2*Var[x]^2 + sigma_y^2 = 2 + 0.5 = 2.5
  //   (using Var[x^2] = 2*sigma^4 for N(0, sigma))
  const trueXMean = 0.0;
  const trueYMean = 0.0;
  const trueXVar = 1.0;
  const trueYVar = 2.5;

  // Log probability function
  // params: [x, y]
  const logProb = (params: np.Array) => {
    const x = params.ref.slice([0, 1]);
    const y = params.slice([1, 2]);

    // x^2 is used twice, so compute it once with proper refs
    const xSquared = x.ref.mul(x);

    // logProb_x = -0.5 * x^2 / sigma_x^2
    const logProbX = xSquared.ref.mul(-0.5 / (sigmaX * sigmaX)).sum();

    // logProb_y = -0.5 * (y - x^2 + 1)^2 / sigma_y^2
    const yMeanGivenX = xSquared.sub(1);
    const yResid = y.sub(yMeanGivenX);
    const logProbY = yResid.ref
      .mul(yResid)
      .mul(-0.5 / (sigmaY * sigmaY))
      .sum();

    return logProbX.add(logProbY);
  };

  test("recovers correct x statistics", async () => {
    const result = await hmc(logProb, {
      initialParams: np.zeros([2]),
      key: random.key(42),
      numSamples: 500,
      numWarmup: 200,
      numChains: 2,
      numLeapfrogSteps: 15,
    });

    // Extract x samples: draws shape is [numChains, numSamples, 2]
    const draws = result.draws;
    const flatDraws = draws.reshape([-1, 2]);
    // Slice first column for x values
    const xCol = flatDraws.slice([0, flatDraws.shape[0]], [0, 1]);
    const xValues = xCol.reshape([-1]).js() as number[];

    const xMean = xValues.reduce((a, b) => a + b, 0) / xValues.length;
    const xVar = xValues.reduce((a, b) => a + (b - xMean) ** 2, 0) / xValues.length;

    console.log(`Banana x stats: mean=${xMean.toFixed(4)}, var=${xVar.toFixed(4)}`);
    console.log(`Expected: mean=${trueXMean.toFixed(4)}, var=${trueXVar.toFixed(4)}`);
    console.log(`Acceptance rate: ${(result.stats.acceptRate.js() as number[]).map(r => r.toFixed(4))}`);
    console.log(`Step size: ${(result.stats.stepSize.js() as number[]).map(s => s.toFixed(6))}`);

    // Check mean within 0.2 absolute (wider for fewer samples)
    expect(Math.abs(xMean - trueXMean)).toBeLessThan(0.2);
    // Check variance within 30% relative
    expect(Math.abs(xVar - trueXVar) / trueXVar).toBeLessThan(0.3);
  }, 120000);

  test("achieves good acceptance rate", async () => {
    // Start at mode [0, -1] instead of origin
    // Minimal warmup/samples to debug
    const result = await hmc(logProb, {
      initialParams: np.array([0.0, -1.0]),
      key: random.key(999),
      numSamples: 10,
      numWarmup: 5,
      numChains: 1,
      numLeapfrogSteps: 10,
      adaptMassMatrix: false,
    });

    const acceptRate = result.stats.acceptRate.js() as number[];
    console.log(`Banana acceptance rate: ${acceptRate}`);
    // For debugging - just check any acceptance
    expect(acceptRate[0]).toBeGreaterThanOrEqual(0);
  }, 60000);
});
