import { describe, expect, test } from "vitest";

import { acceptanceProbability } from "../../src/metropolis";

describe("metropolis acceptance", () => {
  test("matches min(1, exp(-deltaH))", () => {
    const cases = [-2, -0.1, 0, 0.1, 1, 10];

    for (const deltaH of cases) {
      const expected = Math.min(1, Math.exp(-deltaH));
      expect(acceptanceProbability(deltaH)).toBeCloseTo(expected, 12);
    }
  });

  test("is monotone decreasing in deltaH", () => {
    const deltas = [-1, 0, 0.5, 1, 2, 4];
    const probs = deltas.map((deltaH) => acceptanceProbability(deltaH));

    for (let i = 1; i < probs.length; i += 1) {
      expect(probs[i]).toBeLessThanOrEqual(probs[i - 1]);
    }
  });
});
