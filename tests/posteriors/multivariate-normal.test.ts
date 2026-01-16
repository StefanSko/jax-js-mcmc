import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, random, type Array } from "@jax-js/jax";
import { hmc } from "../../src/hmc";
import { rhat } from "../../src/diagnostics";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const trueMean = np.array([0, 0]);
const trueCov = np.array([
  [1, 0.8],
  [0.8, 1],
]);
const invCov = np.linalg.inv(trueCov);

const logProb = (p: { x: Array }) => {
  const diff = p.x.sub(trueMean);
  const quad = np.dot(diff, np.dot(invCov, diff));
  return quad.mul(-0.5);
};

describe("multivariate normal posterior", () => {
  test("recovers true mean within 5%", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });

    const sampleMean = np.mean(result.draws.x, [0, 1]);
    const meanVals = sampleMean.dataSync();
    expect(meanVals[0]).toBeCloseTo(0, { tolerance: 0.05 });
    expect(meanVals[1]).toBeCloseTo(0, { tolerance: 0.05 });
  });

  test("recovers true covariance within 10%", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });

    const draws = result.draws.x.reshape([result.draws.x.shape[0] * result.draws.x.shape[1], 2]);
    const cov = np.cov(draws.transpose());
    const covVals = cov.dataSync();
    expect(covVals[0]).toBeCloseTo(1, { tolerance: 0.1 });
    expect(covVals[1]).toBeCloseTo(0.8, { tolerance: 0.1 });
    expect(covVals[3]).toBeCloseTo(1, { tolerance: 0.1 });
  });

  test("R-hat < 1.01 for converged chains", async () => {
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });
    const rhatVals = rhat(result.draws.x) as number[];
    expect(rhatVals[0]).toBeLessThan(1.01);
    expect(rhatVals[1]).toBeLessThan(1.01);
  });
});
