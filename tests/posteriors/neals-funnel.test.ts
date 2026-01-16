import { describe, expect, test } from "vitest";

import { numpy as np, random, type Array as JaxArray } from "@jax-js/jax";

import { hmc } from "../../src/hmc";

function normalLogPdf(x: JaxArray, mean: number, scale: JaxArray): JaxArray {
  const z = np.divide(np.subtract(x, mean), scale.ref);
  const term = np.multiply(-0.5, np.square(z));
  return np.subtract(term, np.log(scale));
}

describe("Neal's funnel", () => {
  test("samples both neck and mouth of funnel", async () => {
    const logProb = (params: { v: JaxArray; x: JaxArray }) => {
      const v = params.v;
      const x = params.x;
      const logPv = normalLogPdf(v.ref, 0, np.array(3.0));
      const scale = np.exp(np.multiply(0.5, v));
      const logPx = np.sum(normalLogPdf(x.ref, 0, scale));
      return np.add(logPv, logPx);
    };

    const result = await hmc(logProb, {
      numSamples: 150,
      numWarmup: 80,
      numLeapfrogSteps: 8,
      initialParams: { v: np.array(0.0), x: np.array([0.0, 0.0]) },
      key: random.key(5),
    });

    const vSamples = result.draws.v;
    const vMin = np.min(vSamples.ref).item() as number;
    const vMax = np.max(vSamples.ref).item() as number;

    expect(vMin).toBeLessThan(-1.2);
    expect(vMax).toBeGreaterThan(1.2);

    const vMean = np.mean(vSamples.ref).item() as number;
    const vStd = np.std(vSamples).item() as number;

    expect(Math.abs(vMean)).toBeLessThan(0.8);
    expect(Math.abs(vStd - 3)).toBeLessThan(1.5);
  }, 10000);
});
