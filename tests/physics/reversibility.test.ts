import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, Array, tree } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog.js";

/**
 * Reversibility Tests
 *
 * The leapfrog integrator is time-reversible:
 * If we run forward L steps, then negate momentum and run L more steps,
 * we should return to the starting position (up to floating point error).
 *
 * This is a fundamental property required for HMC correctness.
 * Tolerance: 1e-5 (float32-friendly)
 */

describe("leapfrog reversibility", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  /**
   * Gradient of log prob for standard normal: grad(logProb) = -q
   */
  function gradLogProb(q: Array): Array {
    return q.mul(-1) as Array;
  }

  /**
   * Check if two arrays are close within tolerance.
   */
  async function arraysClose(a: Array, b: Array, tolerance: number): Promise<boolean> {
    const diff = a.sub(b).abs();
    const maxDiff = diff.max().item() as number;
    return maxDiff < tolerance;
  }

  test("forward then backward returns to start (1D)", async () => {
    const stepSize = 0.1;
    const numSteps = 50;
    const tolerance = 1e-5;

    const q0 = np.array([1.5], { dtype: np.DType.Float32 });
    const p0 = np.array([0.7], { dtype: np.DType.Float32 });

    // Forward
    const [q1, p1] = leapfrog(q0, p0, gradLogProb, stepSize, numSteps);

    // Backward (negate momentum)
    const p1Neg = (p1 as Array).mul(-1) as Array;
    const [q2, p2] = leapfrog(q1, p1Neg, gradLogProb, stepSize, numSteps);

    // Should return to start
    expect(await arraysClose(q2 as Array, q0, tolerance)).toBe(true);
    expect(await arraysClose((p2 as Array).mul(-1) as Array, p0, tolerance)).toBe(true);
  });

  test("forward then backward returns to start (3D)", async () => {
    const stepSize = 0.1;
    const numSteps = 50;
    const tolerance = 1e-5;

    const q0 = np.array([1.0, 2.0, 3.0], { dtype: np.DType.Float32 });
    const p0 = np.array([0.5, -0.5, 0.1], { dtype: np.DType.Float32 });

    // Forward
    const [q1, p1] = leapfrog(q0, p0, gradLogProb, stepSize, numSteps);

    // Backward (negate momentum)
    const p1Neg = (p1 as Array).mul(-1) as Array;
    const [q2, p2] = leapfrog(q1, p1Neg, gradLogProb, stepSize, numSteps);

    // Should return to start
    expect(await arraysClose(q2 as Array, q0, tolerance)).toBe(true);
    expect(await arraysClose((p2 as Array).mul(-1) as Array, p0, tolerance)).toBe(true);
  });

  test("reversibility with pytree parameters", async () => {
    const stepSize = 0.1;
    const numSteps = 30;
    const tolerance = 1e-5;

    // Pytree parameters
    const q0 = {
      mu: np.array([1.0, -0.5], { dtype: np.DType.Float32 }),
      sigma: np.array([0.5], { dtype: np.DType.Float32 }),
    };
    const p0 = {
      mu: np.array([0.3, -0.2], { dtype: np.DType.Float32 }),
      sigma: np.array([0.1], { dtype: np.DType.Float32 }),
    };

    // Gradient for pytree (standard normal on each leaf)
    function gradLogProbTree(q: typeof q0): typeof q0 {
      return tree.map((x: Array) => x.mul(-1) as Array, q) as typeof q0;
    }

    // Forward
    const [q1, p1] = leapfrog(q0, p0, gradLogProbTree as any, stepSize, numSteps);

    // Backward (negate momentum)
    const p1Neg = tree.map((x: Array) => x.mul(-1) as Array, p1) as typeof p0;
    const [q2, p2] = leapfrog(q1, p1Neg, gradLogProbTree as any, stepSize, numSteps);

    // Should return to start
    const q2Typed = q2 as typeof q0;
    const p2Typed = p2 as typeof p0;

    expect(await arraysClose(q2Typed.mu, q0.mu, tolerance)).toBe(true);
    expect(await arraysClose(q2Typed.sigma, q0.sigma, tolerance)).toBe(true);
    expect(await arraysClose(p2Typed.mu.mul(-1) as Array, p0.mu, tolerance)).toBe(true);
    expect(await arraysClose(p2Typed.sigma.mul(-1) as Array, p0.sigma, tolerance)).toBe(true);
  });

  test("reversibility with different step sizes", async () => {
    const numSteps = 20;
    const tolerance = 1e-5;

    const q0 = np.array([0.8, -1.2], { dtype: np.DType.Float32 });
    const p0 = np.array([0.4, 0.6], { dtype: np.DType.Float32 });

    for (const stepSize of [0.05, 0.1, 0.2]) {
      // Forward
      const [q1, p1] = leapfrog(q0.ref, p0.ref, gradLogProb, stepSize, numSteps);

      // Backward
      const p1Neg = (p1 as Array).mul(-1) as Array;
      const [q2, p2] = leapfrog(q1, p1Neg, gradLogProb, stepSize, numSteps);

      // Should return to start
      expect(await arraysClose(q2 as Array, q0, tolerance)).toBe(true);
      expect(await arraysClose((p2 as Array).mul(-1) as Array, p0, tolerance)).toBe(true);
    }
  });
});
