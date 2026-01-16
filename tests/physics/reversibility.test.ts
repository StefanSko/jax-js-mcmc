import { describe, expect, test } from "vitest";

import { numpy as np, type Array as JaxArray } from "@jax-js/jax";

import { leapfrog } from "../../src/leapfrog";

function gradPotential(q: JaxArray): JaxArray {
  return q;
}

describe("leapfrog reversibility", () => {
  test("forward then backward returns to start", () => {
    const q0 = np.array([1.0, 2.0, 3.0]);
    const p0 = np.array([0.5, -0.5, 0.1]);
    const stepSize = 0.05;
    const numSteps = 100;

    const [q1, p1] = leapfrog(q0.ref, p0.ref, gradPotential, stepSize, numSteps);
    const [q2, p2] = leapfrog(
      q1 as JaxArray,
      np.negative(p1 as JaxArray),
      gradPotential,
      stepSize,
      numSteps,
    );

    const qErr = np.max(np.abs(np.subtract(q2 as JaxArray, q0))).item() as number;
    const pErr = np.max(np.abs(np.subtract(np.negative(p2 as JaxArray), p0))).item() as number;

    expect(qErr).toBeLessThan(1e-6);
    expect(pErr).toBeLessThan(1e-6);
  });
});
