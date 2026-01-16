import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, random, type Array } from "@jax-js/jax";
import { hmc } from "../../src/hmc";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (p: { v: Array; x: Array }) => {
  const v = p.v;
  const x = p.x;
  const logPv = v.pow(2).div(9).mul(-0.5);
  const sigma = np.exp(v.div(2));
  const logPx = x.pow(2).div(sigma.pow(2)).mul(-0.5).sub(np.log(sigma));
  return logPv.sum().add(logPx.sum());
};

describe("Neal's funnel", () => {
  test("samples from both neck and mouth of funnel", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1500,
      numChains: 4,
      initialParams: { v: np.zeros([1]), x: np.zeros([8]) },
      key: random.key(42),
    });

    const vSamples = result.draws.v.flatten();
    expect(np.min(vSamples).item()).toBeLessThan(-3);
    expect(np.max(vSamples).item()).toBeGreaterThan(3);
  });

  test("marginal of v matches Normal(0, 3)", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1500,
      numChains: 4,
      initialParams: { v: np.zeros([1]), x: np.zeros([8]) },
      key: random.key(42),
    });

    const vSamples = result.draws.v.flatten();
    const vMean = np.mean(vSamples).item();
    const vStd = np.std(vSamples).item();

    expect(vMean).toBeCloseTo(0, { tolerance: 0.25 });
    expect(vStd).toBeCloseTo(3, { tolerance: 0.35 });
  });
});
