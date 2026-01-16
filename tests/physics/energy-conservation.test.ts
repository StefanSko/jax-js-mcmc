import { describe, expect, test } from "vitest";

import { numpy as np, type Array as JaxArray } from "@jax-js/jax";

import { leapfrog } from "../../src/leapfrog";

function potential(q: JaxArray): JaxArray {
  return np.multiply(0.5, np.sum(np.square(q)));
}

function kinetic(p: JaxArray): JaxArray {
  return np.multiply(0.5, np.sum(np.square(p)));
}

function gradPotential(q: JaxArray): JaxArray {
  return q;
}

function hamiltonian(q: JaxArray, p: JaxArray): JaxArray {
  return np.add(potential(q), kinetic(p));
}

describe("leapfrog energy conservation", () => {
  test("energy drift bounded by O(L * stepSize^2)", () => {
    const stepSize = 0.1;
    const numSteps = 100;
    const q0 = np.array([1.0, 0.5]);
    const p0 = np.array([0.0, 1.0]);

    const h0 = hamiltonian(q0.ref, p0.ref).item() as number;
    const [q1, p1] = leapfrog(q0.ref, p0.ref, gradPotential, stepSize, numSteps);
    const h1 = hamiltonian(q1 as JaxArray, p1 as JaxArray).item() as number;

    const energyDrift = Math.abs(h1 - h0);
    const theoreticalBound = numSteps * stepSize * stepSize * 1.5;

    expect(energyDrift).toBeLessThan(theoreticalBound);
  });

  test("energy drift scales quadratically with step size", () => {
    const stepSizes = [0.1, 0.05, 0.025];
    const totalTime = 1.0;

    const drifts = stepSizes.map((stepSize) => {
      const numSteps = Math.round(totalTime / stepSize);
      const q0 = np.array([1.0, 0.5]);
      const p0 = np.array([0.0, 1.0]);
      const h0 = hamiltonian(q0.ref, p0.ref).item() as number;
      const [q1, p1] = leapfrog(q0.ref, p0.ref, gradPotential, stepSize, numSteps);
      const h1 = hamiltonian(q1 as JaxArray, p1 as JaxArray).item() as number;
      return Math.abs(h1 - h0);
    });

    const ratio1 = drifts[1] / drifts[0];
    const ratio2 = drifts[2] / drifts[1];

    expect(Math.abs(ratio1 - 0.25)).toBeLessThan(0.2);
    expect(Math.abs(ratio2 - 0.25)).toBeLessThan(0.2);
  });
});
