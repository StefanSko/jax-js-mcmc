import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";
import { maxAbsDiff, negate } from "../utils/vector";

function gradPotential(q: number[]): number[] {
  return q.slice();
}

describe("leapfrog reversibility", () => {
  test("forward then backward returns to start", () => {
    const q0 = [1.0, 2.0, 3.0];
    const p0 = [0.5, -0.5, 0.1];
    const stepSize = 0.05;
    const numSteps = 100;

    const [q1, p1] = leapfrog(q0, p0, gradPotential, stepSize, numSteps);
    const [q2, p2] = leapfrog(q1, negate(p1), gradPotential, stepSize, numSteps);

    const qErr = maxAbsDiff(q2, q0);
    const pErr = maxAbsDiff(negate(p2), p0);

    expect(qErr).toBeLessThan(1e-6);
    expect(pErr).toBeLessThan(1e-6);
  });
});
