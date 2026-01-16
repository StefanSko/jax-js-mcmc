/**
 * Volume Preservation Tests for Leapfrog Integrator
 *
 * Tests that the leapfrog integrator is symplectic (volume-preserving).
 * The Jacobian determinant of the phase space mapping should be 1.
 *
 * This is essential for the detailed balance of HMC.
 */

import { grad, jacfwd, numpy as np } from "@jax-js/jax";
import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";

describe("Leapfrog Volume Preservation", () => {
  test("jacobian determinant is 1 for simple system", () => {
    // For a 2D system (q has 2 elements, p has 2 elements = 4D phase space)
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    const eps = 0.1;
    const numSteps = 5;

    // Create function that maps [q0, q1, p0, p1] -> [q0', q1', p0', p1']
    // Use slice API: x.slice([start, end]) for 1D array
    const leapfrogFlat = (state: np.Array) => {
      // Use .ref because slice consumes its input
      const q = state.ref.slice([0, 2]);
      const p = state.slice([2, 4]);

      const [qOut, pOut] = leapfrog(
        q,
        p,
        (qIn) => gradLogProb(qIn),
        eps,
        numSteps,
        undefined,
      );

      return np.concatenate([qOut, pOut]);
    };

    // Compute Jacobian using forward-mode AD
    const state0 = np.array([1.0, 0.5, 0.3, -0.2]);
    const jac = jacfwd(leapfrogFlat)(state0);

    // Compute determinant
    const detJac = np.linalg.det(jac).js() as number;

    // Determinant should be 1 (symplectic property)
    // Tolerance: 1e-4 (from DESIGN.md)
    expect(Math.abs(detJac - 1.0)).toBeLessThan(1e-4);
  });

  test("jacobian determinant is 1 for different initial conditions", () => {
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    const eps = 0.15;
    const numSteps = 8;

    const leapfrogFlat = (state: np.Array) => {
      // Use .ref because slice consumes its input
      const q = state.ref.slice([0, 2]);
      const p = state.slice([2, 4]);

      const [qOut, pOut] = leapfrog(
        q,
        p,
        (qIn) => gradLogProb(qIn),
        eps,
        numSteps,
        undefined,
      );

      return np.concatenate([qOut, pOut]);
    };

    // Test several different initial states
    const initialStates = [
      np.array([0.0, 0.0, 1.0, 1.0]),
      np.array([2.0, -1.0, 0.5, -0.5]),
      np.array([-0.5, 1.5, -0.3, 0.8]),
    ];

    for (const state0 of initialStates) {
      const jac = jacfwd(leapfrogFlat)(state0.ref);
      const detJac = np.linalg.det(jac).js() as number;
      expect(Math.abs(detJac - 1.0)).toBeLessThan(1e-4);
    }
  });

  test("volume preservation for 1D system", () => {
    // Simpler case: 1D position, 1D momentum = 2D phase space
    // Note: For 1D, the sum() is still needed to return a scalar
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    const eps = 0.1;
    const numSteps = 10;

    const leapfrogFlat = (state: np.Array) => {
      // Use .ref because slice consumes its input
      const q = state.ref.slice([0, 1]);
      const p = state.slice([1, 2]);

      const [qOut, pOut] = leapfrog(
        q,
        p,
        (qIn) => gradLogProb(qIn),
        eps,
        numSteps,
        undefined,
      );

      return np.concatenate([qOut, pOut]);
    };

    const state0 = np.array([1.0, 0.5]);
    const jac = jacfwd(leapfrogFlat)(state0);
    const detJac = np.linalg.det(jac).js() as number;

    expect(Math.abs(detJac - 1.0)).toBeLessThan(1e-5);
  });

  test("volume is preserved across trajectories (empirical)", () => {
    // Alternative test: check that a "cloud" of points doesn't expand/contract
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-1.5).sum();
    const gradLogProb = grad(logProb);

    // Create a small set of nearby initial states
    const center = np.array([0.0, 0.0, 0.0, 0.0]); // [q0, q1, p0, p1]
    const perturbations = [
      np.array([0.01, 0.0, 0.0, 0.0]),
      np.array([0.0, 0.01, 0.0, 0.0]),
      np.array([0.0, 0.0, 0.01, 0.0]),
      np.array([0.0, 0.0, 0.0, 0.01]),
    ];

    // Extract JS values before consuming the arrays
    const dispBefore = perturbations.map((d) => d.ref.js() as number[]);

    const leapfrogFlat = (state: np.Array) => {
      // Use .ref because slice consumes its input
      const q = state.ref.slice([0, 2]);
      const p = state.slice([2, 4]);

      const [qOut, pOut] = leapfrog(
        q,
        p,
        (qIn) => gradLogProb(qIn),
        0.1,
        10,
        undefined,
      );

      return np.concatenate([qOut, pOut]);
    };

    // Map center and perturbed points
    // Use center.ref so center survives for subsequent calls
    const centerFinal = leapfrogFlat(center.ref);
    const perturbedFinal = perturbations.map((d) =>
      leapfrogFlat(center.ref.add(d)),
    );

    // Compute displacement vectors after (dispBefore already computed above)
    const dispAfter = perturbedFinal.map((pf) =>
      pf.sub(centerFinal.ref).js() as number[],
    );

    // Build matrices from displacement vectors (4x4)
    const matBefore = np.array(dispBefore).transpose();
    const matAfter = np.array(dispAfter).transpose();

    // Determinants should be equal (volume preserved)
    const detBefore = np.linalg.det(matBefore).js() as number;
    const detAfter = np.linalg.det(matAfter).js() as number;

    // Ratio should be 1
    const ratio = Math.abs(detAfter / detBefore);
    expect(ratio).toBeWithinRange(0.95, 1.05);
  });
});
