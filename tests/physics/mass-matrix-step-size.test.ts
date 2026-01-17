import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, random, tree, type Array } from "@jax-js/jax";
import { hmc } from "../../src/hmc";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const sigma = np.array([0.02, 1]);
const logProb = (p: { x: Array }) => {
  const x = p.x.ref.add(0);
  const scaled = x.div(sigma.ref);
  return scaled.mul(scaled.ref).mul(-0.5).sum();
};

describe("mass matrix adaptation", () => {
  test("retunes step size after adapting mass matrix", async () => {
    const base = {
      numSamples: 25,
      numWarmup: 80,
      numChains: 1,
      numLeapfrogSteps: 10,
      initialParams: { x: np.zeros([2]) },
    };

    const noAdapt = await hmc(logProb, {
      ...base,
      adaptMassMatrix: false,
      key: random.key(0),
    });

    const withAdapt = await hmc(logProb, {
      ...base,
      adaptMassMatrix: true,
      key: random.key(0),
    });

    const stepSizeDelta = Math.abs(withAdapt.stats.stepSize - noAdapt.stats.stepSize);
    expect(stepSizeDelta / noAdapt.stats.stepSize).toBeGreaterThan(0.2);

    const mass = (withAdapt.stats.massMatrix as { x: Array }).x;
    const massVals = mass.ref.dataSync();
    expect(massVals[0]).toBeLessThan(massVals[1]);

    tree.dispose(noAdapt.draws);
    tree.dispose(withAdapt.draws);
    tree.dispose(noAdapt.stats.massMatrix as any);
    tree.dispose(withAdapt.stats.massMatrix as any);
  });
});
