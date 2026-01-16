import { describe, expect, test } from "vitest";

import { numpy as np, random, type Array as JaxArray } from "@jax-js/jax";

import { hmc } from "../../src/hmc";

const basis1 = np.array([1.0, 0.0]);
const basis2 = np.array([0.0, 1.0]);

function normalLogPdf(x: JaxArray, mean: JaxArray, scale: JaxArray): JaxArray {
  const z = np.divide(np.subtract(x, mean), scale.ref);
  const term = np.multiply(-0.5, np.square(z));
  return np.subtract(term, np.log(scale));
}

describe("banana posterior", () => {
  test("recovers banana-shaped correlation", async () => {
    const b = 0.1;
    const logProb = (params: { x: JaxArray }) => {
      const x1 = np.dot(params.x.ref, basis1.ref);
      const x2 = np.dot(params.x.ref, basis2.ref);
      const logP1 = normalLogPdf(x1.ref, np.array(0.0), np.array(10.0));
      const mean2 = np.multiply(b, np.square(x1));
      const logP2 = normalLogPdf(x2.ref, mean2, np.array(1.0));
      return np.add(logP1, logP2);
    };

    const result = await hmc(logProb, {
      numSamples: 150,
      numWarmup: 80,
      numLeapfrogSteps: 10,
      initialParams: { x: np.array([0.5, 0.5]) },
      key: random.key(7),
    });

    const draws = result.draws.x;
    const shape = np.shape(draws.ref) as number[];
    const total = shape[0] * shape[1];
    const flat = np.reshape(draws.ref, [total, shape[2]]);

    const x1 = np.matmul(flat.ref, basis1.ref);
    const x2 = np.matmul(flat.ref, basis2.ref);
    const corr = np.corrcoef(np.square(x1), x2).js() as number[][];

    expect(corr[0][1]).toBeGreaterThan(0.2);
  }, 20000);
});
