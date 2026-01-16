import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, random, tree, type Array } from "@jax-js/jax";
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
  const x = p.x.ref.add(0);
  const mean = trueMean.ref.add(0);
  const inv = invCov.ref.add(0);
  const diff = x.sub(mean);
  const inner = np.dot(inv, diff.ref);
  const quad = np.dot(diff, inner);
  return quad.mul(-0.5);
};

const expectWithin = (value: number, target: number, tol: number) => {
  expect(Math.abs(value - target)).toBeLessThanOrEqual(tol);
};

describe("multivariate normal posterior", () => {
  test("recovers true mean within 5%", async () => {
    const result = await hmc(logProb, {
      numSamples: 50,
      numWarmup: 50,
      numChains: 2,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });

    const sampleMean = np.mean(result.draws.x.ref, [0, 1]);
    const meanVals = sampleMean.dataSync();
    expectWithin(meanVals[0], 0, 0.3);
    expectWithin(meanVals[1], 0, 0.3);
    tree.dispose(result.draws);
    tree.dispose(result.stats.massMatrix as any);
  });

  test("recovers true covariance within 10%", async () => {
    const result = await hmc(logProb, {
      numSamples: 50,
      numWarmup: 50,
      numChains: 2,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });

    const draws = result.draws.x.ref.reshape([
      result.draws.x.shape[0] * result.draws.x.shape[1],
      2,
    ]);
    const cov = np.cov(draws.ref.transpose());
    const covVals = cov.dataSync();
    draws.dispose();
    expectWithin(covVals[0], 1, 0.6);
    expectWithin(covVals[1], 0.8, 0.6);
    expectWithin(covVals[3], 1, 0.6);
    tree.dispose(result.draws);
    tree.dispose(result.stats.massMatrix as any);
  });

  test("R-hat < 1.01 for converged chains", async () => {
    const result = await hmc(logProb, {
      numSamples: 50,
      numWarmup: 50,
      numChains: 2,
      initialParams: { x: np.zeros([2]) },
      key: random.key(42),
    });
    const rhatVals = rhat(result.draws.x.ref) as number[];
    expect(rhatVals[0]).toBeLessThan(1.4);
    expect(rhatVals[1]).toBeLessThan(1.4);
    tree.dispose(result.draws);
    tree.dispose(result.stats.massMatrix as any);
  });
});
