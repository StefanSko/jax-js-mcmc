import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";

function gradPotential(q: number[]): number[] {
  return q.slice();
}

describe("leapfrog integrator accuracy", () => {
  test("harmonic oscillator matches analytic solution", () => {
    const stepSize = 0.01;
    const numSteps = 100;
    const q0 = [0.0];
    const p0 = [1.0];

    const [q1, p1] = leapfrog(q0, p0, gradPotential, stepSize, numSteps);

    expect(q1[0]).toBeCloseTo(Math.sin(1.0), 2);
    expect(p1[0]).toBeCloseTo(Math.cos(1.0), 2);
  });
});
