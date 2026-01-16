import { describe, expect, test } from "vitest";

import { numpy as np, random, type Array as JaxArray } from "@jax-js/jax";

import { hmc } from "../../src/hmc";
import reference from "./normal-reference.json";

function normalLogProb(x: { value: JaxArray }): JaxArray {
  return np.multiply(-0.5, np.square(x.value));
}

describe("reference comparison", () => {
  test("matches normal reference statistics", async () => {
    const result = await hmc(normalLogProb, {
      numSamples: 600,
      numWarmup: 300,
      numLeapfrogSteps: 15,
      initialParams: { value: np.array(0.5) },
      key: random.key(9),
    });

    const draws = result.draws.value;
    const mean = np.mean(draws.ref).item() as number;
    const sd = Math.sqrt((np.var_(draws).item() as number));

    expect(mean).toBeCloseTo(reference.mean, 0);
    expect(sd).toBeCloseTo(reference.sd, 0);
  }, 20000);
});
