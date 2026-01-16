/**
 * Energy Conservation Tests for Leapfrog Integrator
 *
 * Tests that the leapfrog integrator conserves Hamiltonian energy to O(epsilon^2).
 * This is a fundamental property of symplectic integrators.
 *
 * The Hamiltonian is: H(q, p) = U(q) + K(p)
 * where U(q) = -logProb(q) (potential energy)
 * and K(p) = 0.5 * p^T * M^{-1} * p (kinetic energy)
 */

import { grad, numpy as np, tree } from "@jax-js/jax";
import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";

describe("Leapfrog Energy Conservation", () => {
  /**
   * Helper to compute Hamiltonian H = U(q) + K(p) = -logProb(q) + 0.5*p^2
   */
  function hamiltonian(
    q: np.Array,
    p: np.Array,
    logProb: (q: np.Array) => np.Array,
  ): np.Array {
    const U = logProb(q.ref).mul(-1);
    const K = p.ref.mul(p).mul(0.5).sum();
    return U.add(K);
  }

  test("energy drift scales as O(epsilon^2)", () => {
    // Simple quadratic potential: U(q) = 0.5 * q^2, so logProb = -0.5 * q^2
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    // Initial state
    const q0 = np.array([1.0, 2.0]);
    const p0 = np.array([0.5, -0.3]);

    const H0 = hamiltonian(q0.ref, p0.ref, logProb).js() as number;

    // Test multiple step sizes
    const stepSizes = [0.1, 0.05, 0.025];
    const energyDrifts: number[] = [];

    for (const eps of stepSizes) {
      const [qFinal, pFinal] = leapfrog(
        q0.ref,
        p0.ref,
        (q) => gradLogProb(q),
        eps,
        10, // numSteps
        undefined, // no mass matrix
      );

      const H1 = hamiltonian(qFinal, pFinal, logProb).js() as number;
      const drift = Math.abs(H1 - H0);
      energyDrifts.push(drift);
    }

    // Energy drift should scale as O(eps^2)
    // Check: drift(eps/2) / drift(eps) ~ 0.25 (quadratic scaling)
    const ratio1 = energyDrifts[1] / energyDrifts[0];
    const ratio2 = energyDrifts[2] / energyDrifts[1];

    // Tolerance: 0.25 +/- 0.2 for float32
    expect(ratio1).toBeWithinRange(0.05, 0.45);
    expect(ratio2).toBeWithinRange(0.05, 0.45);
  });

  test("energy is conserved for small step sizes", () => {
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    const q0 = np.array([1.0, 2.0, -0.5]);
    const p0 = np.array([0.3, -0.2, 0.1]);

    const H0 = hamiltonian(q0.ref, p0.ref, logProb).js() as number;

    const [qFinal, pFinal] = leapfrog(
      q0.ref,
      p0.ref,
      (q) => gradLogProb(q),
      0.01, // very small step size
      50,
      undefined,
    );

    const H1 = hamiltonian(qFinal, pFinal, logProb).js() as number;

    // For small eps, energy drift should be negligible
    expect(Math.abs(H1 - H0)).toBeLessThan(1e-3);
  });

  test("energy conservation with pytree parameters", () => {
    // Test with nested parameter structure
    const logProb = (params: { x: np.Array; y: np.Array }) => {
      const termX = params.x.ref.mul(params.x).mul(-0.5).sum();
      const termY = params.y.ref.mul(params.y).mul(-1.0).sum();
      return termX.add(termY);
    };

    // Compute gradient manually for each component
    const gradLogProb = (params: { x: np.Array; y: np.Array }) => ({
      x: params.x.mul(-1), // d/dx of -0.5*x^2 = -x
      y: params.y.mul(-2), // d/dy of -1.0*y^2 = -2y
    });

    const q0 = { x: np.array([1.0, -0.5]), y: np.array([2.0]) };
    const p0 = { x: np.array([0.3, 0.1]), y: np.array([-0.4]) };

    // Compute initial Hamiltonian
    const U0 = logProb(tree.ref(q0)).mul(-1).js() as number;
    const K0 =
      (tree
        .leaves(tree.ref(p0))
        .map((p) => p.ref.mul(p).mul(0.5).sum().js() as number)
        .reduce((a, b) => a + b, 0) as number);
    const H0 = U0 + K0;

    const [qFinal, pFinal] = leapfrog(
      tree.ref(q0),
      tree.ref(p0),
      gradLogProb,
      0.02,
      25,
      undefined,
    );

    // Compute final Hamiltonian
    const U1 = logProb(tree.ref(qFinal)).mul(-1).js() as number;
    const K1 =
      (tree
        .leaves(tree.ref(pFinal))
        .map((p) => p.ref.mul(p).mul(0.5).sum().js() as number)
        .reduce((a, b) => a + b, 0) as number);
    const H1 = U1 + K1;

    // Check energy conservation
    expect(Math.abs(H1 - H0)).toBeLessThan(1e-2);
  });
});
