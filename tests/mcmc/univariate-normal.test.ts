import { describe, expect, test } from "vitest";

import { numpy as np, random, type Array as JaxArray } from "@jax-js/jax";

import { hmc } from "../../src/hmc";

function normalLogProb(x: { value: JaxArray }): JaxArray {
  return np.multiply(-0.5, np.square(x.value));
}

describe("HMC on N(0,1)", () => {
  test("matches mean and variance", async () => {
    const result = await hmc(normalLogProb, {
      numSamples: 300,
      numWarmup: 200,
      numLeapfrogSteps: 10,
      initialParams: { value: np.array(1.0) },
      key: random.key(0),
    });

    const draws = result.draws.value;
    const mean = np.mean(draws.ref).item() as number;
    const variance = np.var_(draws).item() as number;

    expect(mean).toBeCloseTo(0, 0);
    expect(variance).toBeCloseTo(1, 0);
  }, 20000);
});
