import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, grad, Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog.js";

/**
 * Energy Conservation Tests
 *
 * The leapfrog integrator should conserve the Hamiltonian up to O(ε²) per step.
 * For L steps with step size ε, total energy drift should be bounded by O(L·ε²).
 *
 * The Hamiltonian is: H(q, p) = U(q) + K(p)
 * where U(q) = -logProb(q) is potential energy
 * and K(p) = 0.5 * p^T * M^{-1} * p is kinetic energy
 */

describe("leapfrog energy conservation", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  /**
   * Compute Hamiltonian for a simple quadratic potential.
   * U(q) = 0.5 * q^T * q (standard normal)
   * K(p) = 0.5 * p^T * p (identity mass matrix)
   * Note: Consumes q and p
   */
  function hamiltonian(q: Array, p: Array): number {
    // Use ref for the first operand since mul consumes both
    const U = q.ref.mul(q).mul(0.5).sum();
    const K = p.ref.mul(p).mul(0.5).sum();
    return (U.add(K).item() as number);
  }

  /**
   * Gradient of log prob for standard normal: grad(logProb) = -q
   * Since U(q) = -logProb(q) = 0.5 * q^T * q, grad(U) = q, so grad(logProb) = -q
   */
  function gradLogProb(q: Array): Array {
    return q.mul(-1) as Array;
  }

  test("energy drift bounded by O(L * ε²)", async () => {
    const stepSize = 0.1;
    const numSteps = 100;

    const q0 = np.array([1.0, 0.5], { dtype: np.DType.Float32 });
    const p0 = np.array([0.0, 1.0], { dtype: np.DType.Float32 });

    const H0 = hamiltonian(q0.ref, p0.ref);
    const [q1, p1] = leapfrog(q0, p0, gradLogProb, stepSize, numSteps);
    const H1 = hamiltonian(q1 as Array, p1 as Array);

    const energyDrift = Math.abs(H1 - H0);

    // For leapfrog with L steps, energy error is O(L * ε²)
    // With ε=0.1, L=100, bound is roughly 100 * 0.01 = 1.0
    // In practice it's much smaller, use conservative bound
    const theoreticalBound = numSteps * stepSize * stepSize * 5; // C=5 is conservative
    expect(energyDrift).toBeLessThan(theoreticalBound);
  });

  test("energy drift scales quadratically with step size", async () => {
    const numSteps = 50;

    function measureEnergyDrift(stepSize: number): number {
      // Create fresh arrays for each measurement
      const q0 = np.array([1.0, 0.5, -0.3], { dtype: np.DType.Float32 });
      const p0 = np.array([0.2, -0.4, 0.6], { dtype: np.DType.Float32 });
      const H0 = hamiltonian(q0.ref, p0.ref);
      const [q1, p1] = leapfrog(q0, p0, gradLogProb, stepSize, numSteps);
      const H1 = hamiltonian(q1 as Array, p1 as Array);
      return Math.abs(H1 - H0);
    }

    const stepSizes = [0.1, 0.05, 0.025];
    const drifts = stepSizes.map(measureEnergyDrift);

    // Halving step size should quarter the drift (quadratic scaling)
    // Allow tolerance ±0.2 around expected ratio of 0.25
    const ratio1 = drifts[1]! / drifts[0]!;
    const ratio2 = drifts[2]! / drifts[1]!;

    expect(ratio1).toBeGreaterThan(0.05); // 0.25 - 0.2
    expect(ratio1).toBeLessThan(0.45);    // 0.25 + 0.2
    expect(ratio2).toBeGreaterThan(0.05);
    expect(ratio2).toBeLessThan(0.45);
  });

  test("energy conserved for harmonic oscillator (exact solution)", async () => {
    // For harmonic oscillator with ω=1, energy is exactly conserved
    // Use small step size and many steps to verify long-term stability

    const stepSize = 0.05;
    const numSteps = 200;

    const q0 = np.array([1.0, 0.0], { dtype: np.DType.Float32 });
    const p0 = np.array([0.0, 1.0], { dtype: np.DType.Float32 });

    const H0 = hamiltonian(q0.ref, p0.ref);
    const [q1, p1] = leapfrog(q0, p0, gradLogProb, stepSize, numSteps);
    const H1 = hamiltonian(q1 as Array, p1 as Array);

    const energyDrift = Math.abs(H1 - H0);
    // With 200 steps at 0.05, expect drift < 200 * 0.0025 * 5 = 2.5
    expect(energyDrift).toBeLessThan(2.5);
  });
});
