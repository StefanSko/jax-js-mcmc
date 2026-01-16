import { describe, expect, test } from "vitest";

import { numpy as np, random, type Array as JaxArray } from "@jax-js/jax";

import { hmc } from "../../src/hmc";

const invCovMatrix = np.array([
  [2.7777778, -2.2222222],
  [-2.2222222, 2.7777778],
]);

function mvnLogProb(params: { x: JaxArray }): JaxArray {
  const xCol = np.reshape(params.x, [2, 1]);
  const xRow = np.transpose(xCol.ref);
  const quad = np.matmul(np.matmul(xRow, invCovMatrix.ref), xCol);
  return np.multiply(-0.5, np.squeeze(quad));
}

describe("multivariate normal posterior", () => {
  test("recovers mean and covariance", async () => {
    const result = await hmc(mvnLogProb, {
      numSamples: 300,
      numWarmup: 150,
      numLeapfrogSteps: 15,
      numChains: 1,
      initialParams: { x: np.array([1.0, -1.0]) },
      key: random.key(3),
    });

    const draws = result.draws.x;
    const mean = np.mean(draws.ref, [0, 1]);
    const meanValues = mean.js() as number[];

    expect(meanValues[0]).toBeCloseTo(0, 0);
    expect(meanValues[1]).toBeCloseTo(0, 0);

    const shape = np.shape(draws.ref) as number[];
    const total = shape[0] * shape[1];
    const flat = np.reshape(draws.ref, [total, shape[2]]);
    const cov = np.cov(flat, null, { rowvar: false }).js() as number[][];

    expect(cov[0][0]).toBeCloseTo(1, 0);
    expect(cov[1][1]).toBeCloseTo(1, 0);
    expect(cov[0][1]).toBeCloseTo(0.8, 0);
  }, 10000);
});
