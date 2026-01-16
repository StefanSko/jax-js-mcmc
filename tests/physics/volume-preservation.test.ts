import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";

function gradPotential(q: number[]): number[] {
  return q.slice();
}

describe("leapfrog volume preservation", () => {
  test("jacobian determinant is near 1", () => {
    const stepSize = 0.1;
    const numSteps = 10;
    const eps = 1e-6;

    const f = (q: number, p: number): [number, number] => {
      const [q1, p1] = leapfrog([q], [p], gradPotential, stepSize, numSteps);
      return [q1[0], p1[0]];
    };

    const q0 = 0.2;
    const p0 = -0.3;
    const base = f(q0, p0);
    const dq = f(q0 + eps, p0);
    const dp = f(q0, p0 + eps);

    const dfdq = [(dq[0] - base[0]) / eps, (dq[1] - base[1]) / eps];
    const dfdp = [(dp[0] - base[0]) / eps, (dp[1] - base[1]) / eps];

    const det = dfdq[0] * dfdp[1] - dfdq[1] * dfdp[0];

    expect(Math.abs(det - 1)).toBeLessThan(1e-3);
  });
});
