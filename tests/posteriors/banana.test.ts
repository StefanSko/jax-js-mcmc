import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, random, type Array } from "@jax-js/jax";
import { hmc } from "../../src/hmc";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const b = 0.1;
const logProb = (p: { x: Array }) => {
  const x = p.x;
  const x1 = np.split(x, 2, 0)[0];
  const x2 = np.split(x, 2, 0)[1];
  const logPx1 = x1.pow(2).div(100).mul(-0.5);
  const mean = x1.pow(2).mul(b);
  const logPx2 = x2.sub(mean).pow(2).mul(-0.5);
  return logPx1.add(logPx2).sum();
};

describe("banana posterior", () => {
  test("recovers curved posterior shape", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });

    const chains = result.draws.x.shape[0];
    const samples = result.draws.x.shape[1];
    const [x1Raw, x2Raw] = np.split(result.draws.x, 2, 2);
    const x1 = x1Raw.reshape([chains, samples]);
    const x2 = x2Raw.reshape([chains, samples]);
    const x1Squared = x1.pow(2);

    const corr = np.corrcoef(x1Squared.flatten(), x2.flatten());
    const corrVal = corr.dataSync()[1];
    expect(corrVal).toBeGreaterThan(0.8);
  });
});
