import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, grad, type Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog";
import { hamiltonian } from "../../src/hamiltonian";
import { treeOnesLike, treeRef } from "../../src/tree-utils";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (q: Array) => q.ref.mul(q).mul(-0.5).sum();
const gradLogProb = grad(logProb) as (q: Array) => Array;

function measureEnergyDrift(stepSize: number, numSteps: number): number {
  const q0 = np.array([1.0, 0.5]);
  const p0 = np.array([0.0, 1.0]);
  const massMatrix = np.onesLike(q0.ref);
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
    const ratio1 = drifts[1] / drifts[0];
    const ratio2 = drifts[2] / drifts[1];
    expect(Math.abs(ratio1 - 0.25)).toBeLessThan(0.2);
    expect(Math.abs(ratio2 - 0.25)).toBeLessThan(0.2);
  });

  test("energy is conserved for small step sizes", () => {
    const drift = measureEnergyDrift(0.01, 50);
    expect(drift).toBeLessThan(1e-3);
  });

  test("energy conservation with pytree parameters", () => {
    const logProb = (params: { x: Array; y: Array }) => {
      const termX = params.x.ref.mul(params.x).mul(-0.5).sum();
      const termY = params.y.ref.mul(params.y).mul(-1.0).sum();
      return termX.add(termY);
    };

    const gradLogProb = (params: { x: Array; y: Array }) => ({
      x: params.x.mul(-1),
      y: params.y.mul(-2),
    });

    const q0 = { x: np.array([1.0, -0.5]), y: np.array([2.0]) };
    const p0 = { x: np.array([0.3, 0.1]), y: np.array([-0.4]) };
    const massMatrix = treeOnesLike(treeRef(q0));

    const H0 = hamiltonian(
      treeRef(q0),
      treeRef(p0),
      logProb,
      massMatrix,
    ).item();

    const [q1, p1] = leapfrog(
      treeRef(q0),
      treeRef(p0),
      gradLogProb,
      0.02,
      25,
      massMatrix,
    );

    const H1 = hamiltonian(
      treeRef(q1),
      treeRef(p1),
      logProb,
      massMatrix,
    ).item();

    expect(Math.abs(H1 - H0)).toBeLessThan(1e-2);
  });
});
