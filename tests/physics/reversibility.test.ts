/**
 * Reversibility Tests for Leapfrog Integrator
 *
 * Tests that the leapfrog integrator is time-reversible:
 * Running forward integration, then negating momentum, then running
 * forward integration again should return to the starting point.
 *
 * This is a key property for detailed balance in HMC.
 */

import { grad, numpy as np, tree } from "@jax-js/jax";
import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";

describe("Leapfrog Reversibility", () => {
  test("forward then backward integration returns to start", () => {
    // Use a non-trivial potential
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    const q0 = np.array([1.0, -0.5, 2.0]);
    const p0 = np.array([0.3, 0.1, -0.4]);

    const eps = 0.1;
    const numSteps = 20;

    // Forward integration
    const [q1, p1] = leapfrog(
      q0.ref,
      p0.ref,
      (q) => gradLogProb(q),
      eps,
      numSteps,
      undefined,
    );

    // Backward integration: negate momentum, integrate, negate momentum again
    const p1Neg = p1.mul(-1);
    const [q2, p2Neg] = leapfrog(
      q1.ref,
      p1Neg,
      (q) => gradLogProb(q),
      eps,
      numSteps,
      undefined,
    );
    const p2 = p2Neg.mul(-1);

    // Check q2 ~ q0 and p2 ~ p0
    const qDiff = np.max(np.abs(q2.sub(q0))).js() as number;
    const pDiff = np.max(np.abs(p2.sub(p0))).js() as number;

    // Tolerance: 1e-5 (from DESIGN.md)
    expect(qDiff).toBeLessThan(1e-5);
    expect(pDiff).toBeLessThan(1e-5);
  });

  test("single step reversibility", () => {
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    const q0 = np.array([1.5, -2.0, 0.7]);
    const p0 = np.array([0.5, 0.3, -0.2]);

    // Forward one step
    const [q1, p1] = leapfrog(
      q0.ref,
      p0.ref,
      (q) => gradLogProb(q),
      0.2,
      1, // single step
      undefined,
    );

    // Backward one step (negate momentum)
    const [q2, p2Neg] = leapfrog(
      q1.ref,
      p1.mul(-1),
      (q) => gradLogProb(q),
      0.2,
      1,
      undefined,
    );
    const p2 = p2Neg.mul(-1);

    const qDiff = np.max(np.abs(q2.sub(q0))).js() as number;
    const pDiff = np.max(np.abs(p2.sub(p0))).js() as number;

    expect(qDiff).toBeLessThan(1e-5);
    expect(pDiff).toBeLessThan(1e-5);
  });

  test("reversibility with pytree parameters", () => {
    // Test with nested parameter structure
    const gradLogProb = (params: { x: np.Array; y: np.Array }) => ({
      x: params.x.mul(-1), // gradient of -0.5*x^2
      y: params.y.mul(-2), // gradient of -1.0*y^2
    });

    const q0 = { x: np.array([1.0, -0.5]), y: np.array([2.0]) };
    const p0 = { x: np.array([0.3, 0.1]), y: np.array([-0.4]) };

    const eps = 0.15;
    const numSteps = 15;

    // Forward integration
    const [q1, p1] = leapfrog(
      tree.ref(q0),
      tree.ref(p0),
      gradLogProb,
      eps,
      numSteps,
      undefined,
    );

    // Backward integration: negate momentum
    const p1Neg = tree.map((p: np.Array) => p.mul(-1), tree.ref(p1));
    const [q2, p2Neg] = leapfrog(
      tree.ref(q1),
      p1Neg,
      gradLogProb,
      eps,
      numSteps,
      undefined,
    );
    const p2 = tree.map((p: np.Array) => p.mul(-1), p2Neg);

    // Check q2 ~ q0 and p2 ~ p0 for each component
    const qDiffX = np.max(np.abs(q2.x.sub(q0.x))).js() as number;
    const qDiffY = np.max(np.abs(q2.y.sub(q0.y))).js() as number;
    const pDiffX = np.max(np.abs(p2.x.sub(p0.x))).js() as number;
    const pDiffY = np.max(np.abs(p2.y.sub(p0.y))).js() as number;

    expect(qDiffX).toBeLessThan(1e-5);
    expect(qDiffY).toBeLessThan(1e-5);
    expect(pDiffX).toBeLessThan(1e-5);
    expect(pDiffY).toBeLessThan(1e-5);
  });

  test("reversibility with non-quadratic potential", () => {
    // More challenging: quartic potential
    // logProb = -0.25 * x^4 = -0.25 * (x^2)^2
    const logProb = (q: np.Array) => {
      const qSquared = q.ref.mul(q); // q^2
      return qSquared.ref.mul(qSquared).mul(-0.25).sum(); // (q^2)^2
    };
    const gradLogProb = grad(logProb);

    const q0 = np.array([0.8, -0.6]);
    const p0 = np.array([0.2, 0.3]);

    const eps = 0.05; // Smaller step for harder potential
    const numSteps = 30;

    // Forward
    const [q1, p1] = leapfrog(
      q0.ref,
      p0.ref,
      (q) => gradLogProb(q),
      eps,
      numSteps,
      undefined,
    );

    // Backward
    const [q2, p2Neg] = leapfrog(
      q1.ref,
      p1.mul(-1),
      (q) => gradLogProb(q),
      eps,
      numSteps,
      undefined,
    );
    const p2 = p2Neg.mul(-1);

    const qDiff = np.max(np.abs(q2.sub(q0))).js() as number;
    const pDiff = np.max(np.abs(p2.sub(p0))).js() as number;

    expect(qDiff).toBeLessThan(1e-4); // Slightly looser for harder potential
    expect(pDiff).toBeLessThan(1e-4);
  });
});
