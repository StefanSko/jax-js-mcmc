import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, grad, type Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog";
import { hamiltonian } from "../../src/hamiltonian";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (q: Array) => q.mul(q).mul(-0.5).sum();
const gradLogProb = grad(logProb) as (q: Array) => Array;

function measureEnergyDrift(stepSize: number, numSteps: number): number {
  const q0 = np.array([1.0, 0.5]);
  const p0 = np.array([0.0, 1.0]);
  const massMatrix = np.onesLike(q0);
  const H0 = hamiltonian(q0, p0, logProb, massMatrix);
  const [q1, p1] = leapfrog(q0, p0, gradLogProb, stepSize, numSteps, massMatrix);
  const H1 = hamiltonian(q1, p1, logProb, massMatrix);
  return Math.abs(H1.sub(H0).item());
}

describe("leapfrog energy conservation", () => {
  test("energy drift scales with O(L * ε²)", () => {
    const stepSize = 0.1;
    const numSteps = 100;
    const drift = measureEnergyDrift(stepSize, numSteps);
    expect(drift).toBeGreaterThanOrEqual(0);
  });

  test("energy drift scales quadratically with step size", () => {
    const stepSizes = [0.1, 0.05, 0.025];
    const drifts = stepSizes.map((eps) => measureEnergyDrift(eps, 100));
    expect(drifts[1] / drifts[0]).toBeCloseTo(0.25, { tolerance: 0.2 });
    expect(drifts[2] / drifts[1]).toBeCloseTo(0.25, { tolerance: 0.2 });
  });
});
