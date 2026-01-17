import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, random, tree, type Array } from "@jax-js/jax";
import { hmc } from "../../src/hmc";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (p: { v: Array; x: Array }) => {
  const v = p.v.ref.add(0);
  const x = p.x.ref.add(0);
  const sigma = np.exp(v.ref.div(2));
  const logPv = v.mul(v.ref).div(9).mul(-0.5);
  const sigmaLog = np.log(sigma.ref);
  const sigmaSq = sigma.mul(sigma.ref);
  const xSq = x.mul(x.ref);
  const logPx = xSq.div(sigmaSq).mul(-0.5).sub(sigmaLog);
  return logPv.sum().add(logPx.sum());
};

describe("Neal's funnel", () => {
  test("samples vary across v", async () => {
    const result = await hmc(logProb, {
      numSamples: 50,
      numWarmup: 50,
      numChains: 2,
      numLeapfrogSteps: 5,
      initialStepSize: 0.001,
      adaptMassMatrix: false,
      initialParams: { v: np.zeros([1]), x: np.zeros([8]) },
      key: random.key(42),
    });

    const vSamples = result.draws.v.ref.flatten();
    const vVals = vSamples.dataSync();
    const vMin = Math.min(...vVals);
    const vMax = Math.max(...vVals);
    expect(Number.isFinite(vMin)).toBe(true);
    expect(Number.isFinite(vMax)).toBe(true);
    expect(vMax - vMin).toBeGreaterThan(0.1);
    tree.dispose(result.draws);
    tree.dispose(result.stats.massMatrix as any);
  });

  test("v moments are in a plausible range", async () => {
    const result = await hmc(logProb, {
      numSamples: 50,
      numWarmup: 50,
      numChains: 2,
      numLeapfrogSteps: 5,
      initialStepSize: 0.001,
      adaptMassMatrix: false,
      initialParams: { v: np.zeros([1]), x: np.zeros([8]) },
      key: random.key(42),
    });

    const vSamples = result.draws.v.ref.flatten();
    const vMean = np.mean(vSamples.ref).item();
    const vStd = np.std(vSamples).item();

    expect(Math.abs(vMean - 0)).toBeLessThanOrEqual(2.5);
    expect(Math.abs(vStd - 3)).toBeLessThanOrEqual(2.0);
    tree.dispose(result.draws);
    tree.dispose(result.stats.massMatrix as any);
  });
});
