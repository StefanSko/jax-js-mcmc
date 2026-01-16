import { describe, expect, test } from "vitest";

import { numpy as np, type Array as JaxArray } from "@jax-js/jax";

import { leapfrog } from "../../src/leapfrog";

function gradPotential(q: JaxArray): JaxArray {
  return q;
}

describe("leapfrog integrator accuracy", () => {
  test("harmonic oscillator matches analytic solution", () => {
    const stepSize = 0.01;
    const numSteps = 100;
    const q0 = np.array([0.0]);
    const p0 = np.array([1.0]);

    const [q1, p1] = leapfrog(q0.ref, p0.ref, gradPotential, stepSize, numSteps);

    const qVal = (q1 as JaxArray).item() as number;
    const pVal = (p1 as JaxArray).item() as number;

    expect(qVal).toBeCloseTo(Math.sin(1.0), 2);
    expect(pVal).toBeCloseTo(Math.cos(1.0), 2);
  });
});
