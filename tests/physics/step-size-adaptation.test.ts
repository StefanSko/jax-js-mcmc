import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, random, tree, type Array } from "@jax-js/jax";
import { hmc } from "../../src/hmc";
import { initDualAverage, updateDualAverage } from "../../src/adaptation";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (p: { x: Array }) => p.x.mul(p.x.ref).mul(-0.5).sum();

describe("step-size adaptation", () => {
  test("dual averaging moves step size toward target", () => {
    const state = initDualAverage(0.1);
    const higherAccept = updateDualAverage(state, 0.95, 0.8);
    const lowerAccept = updateDualAverage(state, 0.1, 0.8);

    expect(higherAccept.logStepSize).toBeGreaterThan(lowerAccept.logStepSize);
  });

  test("smaller step size yields higher accept rate", async () => {
    const stiffLogProb = (p: { x: Array }) => {
      const z = p.x.mul(10);
      return z.mul(z.ref).mul(-0.5).sum();
    };

    const base = {
      numSamples: 50,
      numWarmup: 0,
      numChains: 1,
      numLeapfrogSteps: 15,
      adaptMassMatrix: false,
      initialParams: { x: np.array([0.5]) },
    };

    const small = await hmc(stiffLogProb, {
      ...base,
      initialStepSize: 0.02,
      key: random.key(0),
    });

    const large = await hmc(stiffLogProb, {
      ...base,
      initialStepSize: 0.5,
      key: random.key(1),
    });

    expect(small.stats.acceptRate).toBeGreaterThan(large.stats.acceptRate);
    expect(small.stats.acceptRate).toBeGreaterThan(0);
    expect(large.stats.acceptRate).toBeLessThan(1);

    tree.dispose(small.draws);
    tree.dispose(large.draws);
    tree.dispose(small.stats.massMatrix as any);
    tree.dispose(large.stats.massMatrix as any);
  });
});
