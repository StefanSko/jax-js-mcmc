import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";
import { dot } from "../utils/vector";

function potential(q: number[]): number {
  return 0.5 * dot(q, q);
}

function kinetic(p: number[]): number {
  return 0.5 * dot(p, p);
}

function gradPotential(q: number[]): number[] {
  return q.slice();
}

function hamiltonian(q: number[], p: number[]): number {
  return potential(q) + kinetic(p);
}

describe("leapfrog energy conservation", () => {
  test("energy drift bounded by O(L * stepSize^2)", () => {
    const stepSize = 0.1;
    const numSteps = 100;
    const q0 = [1.0, 0.5];
    const p0 = [0.0, 1.0];

    const h0 = hamiltonian(q0, p0);
    const [q1, p1] = leapfrog(q0, p0, gradPotential, stepSize, numSteps);
    const h1 = hamiltonian(q1, p1);

    const energyDrift = Math.abs(h1 - h0);
    const theoreticalBound = numSteps * stepSize * stepSize * 1.5;

    expect(energyDrift).toBeLessThan(theoreticalBound);
  });

  test("energy drift scales quadratically with step size", () => {
    const stepSizes = [0.1, 0.05, 0.025];
    const totalTime = 1.0;

    const drifts = stepSizes.map((stepSize) => {
      const numSteps = Math.round(totalTime / stepSize);
      const q0 = [1.0, 0.5];
      const p0 = [0.0, 1.0];
      const h0 = hamiltonian(q0, p0);
      const [q1, p1] = leapfrog(q0, p0, gradPotential, stepSize, numSteps);
      const h1 = hamiltonian(q1, p1);
      return Math.abs(h1 - h0);
    });

    const ratio1 = drifts[1] / drifts[0];
    const ratio2 = drifts[2] / drifts[1];

    expect(Math.abs(ratio1 - 0.25)).toBeLessThan(0.2);
    expect(Math.abs(ratio2 - 0.25)).toBeLessThan(0.2);
  });
});
