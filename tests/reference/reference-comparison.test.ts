import { beforeAll, describe, test, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defaultDevice, init, numpy as np, random, type Array } from "@jax-js/jax";
import { hmc } from "../../src/hmc";
import { summary } from "../../src/diagnostics";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const refPath = path.join(__dirname, "eight-schools-reference.json");

const y = np.array([28, 8, -3, 7, -1, 1, 18, 12]);
const sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18]);

const logProb = (p: { mu: Array; logTau: Array; eta: Array }) => {
  const mu = p.mu;
  const tau = np.exp(p.logTau);
  const theta = mu.add(tau.mul(p.eta));

  const logLik = theta
    .sub(y)
    .pow(2)
    .div(sigma.pow(2))
    .mul(-0.5)
    .sub(np.log(sigma))
    .sum();

  const logPriorMu = mu.pow(2).div(25).mul(-0.5).sum();
  const tauScale = 5;
  const normalizer = np.array(2 / (Math.PI * tauScale));
  const denom = np.array(1).add(tau.div(tauScale).pow(2));
  const logPriorTau = np.log(normalizer).sub(np.log(denom)).sum();
  const logPriorEta = p.eta.pow(2).mul(-0.5).sum();

  // Jacobian for tau = exp(logTau)
  const logJac = p.logTau.sum();

  return logLik.add(logPriorMu).add(logPriorTau).add(logPriorEta).add(logJac);
};

describe("blue/green: eight schools vs NumPyro", () => {
  const hasReference = fs.existsSync(refPath);

  const testFn = hasReference ? test : test.skip;

  testFn("mu matches NumPyro reference within 15%", async () => {
    const reference = JSON.parse(fs.readFileSync(refPath, "utf-8"));
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      key: random.key(42),
      initialParams: {
        mu: np.zeros([1]),
        logTau: np.zeros([1]),
        eta: np.zeros([8]),
      },
    });

    const tau = np.exp(result.draws.logTau);
    const stats = summary({ mu: result.draws.mu, tau });

    expect(stats.mu.mean).toBeCloseTo(reference.mu.mean, { tolerance: 0.15 });
    expect(stats.mu.sd).toBeCloseTo(reference.mu.std, { tolerance: 0.15 });
  });

  testFn("tau matches NumPyro reference within 15%", async () => {
    const reference = JSON.parse(fs.readFileSync(refPath, "utf-8"));
    const result = await hmc(logProb, {
      numSamples: 2000,
      numWarmup: 1000,
      numChains: 4,
      key: random.key(42),
      initialParams: {
        mu: np.zeros([1]),
        logTau: np.zeros([1]),
        eta: np.zeros([8]),
      },
    });

    const tau = np.exp(result.draws.logTau);
    const stats = summary({ mu: result.draws.mu, tau });

    expect(stats.tau.mean).toBeCloseTo(reference.tau.mean, { tolerance: 0.15 });
    expect(stats.tau.sd).toBeCloseTo(reference.tau.std, { tolerance: 0.15 });
  });
});
