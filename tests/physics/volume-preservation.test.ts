import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog.js";

/**
 * Volume Preservation Tests
 *
 * The leapfrog integrator is symplectic - it preserves phase space volume.
 * This means the Jacobian determinant of the leapfrog map equals 1.
 *
 * We verify this numerically by computing the Jacobian via finite differences
 * and checking that det(J) ≈ 1.
 *
 * Tolerance: 1e-4 (float32-friendly)
 */

describe("leapfrog volume preservation", () => {
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
   * Compute Jacobian of leapfrog map numerically using finite differences.
   * The leapfrog map takes (q, p) -> (q', p') so the Jacobian is 2n x 2n.
   */
  function computeJacobian(
    q0: number[],
    p0: number[],
    stepSize: number,
    numSteps: number,
    eps: number = 1e-4,
  ): number[][] {
    const n = q0.length;
    const jacobian: number[][] = [];

    // Leapfrog map: (q, p) -> (q', p')
    function leapfrogMap(qp: number[]): number[] {
      const q = np.array(qp.slice(0, n), { dtype: np.DType.Float32 });
      const p = np.array(qp.slice(n), { dtype: np.DType.Float32 });
      const [qNew, pNew] = leapfrog(q, p, gradLogProb, stepSize, numSteps);
      const qArr = (qNew as Array).js() as number[];
      const pArr = (pNew as Array).js() as number[];
      return [...qArr, ...pArr];
    }

    const qp0 = [...q0, ...p0];
    const f0 = leapfrogMap(qp0);

    // Compute Jacobian column by column using central differences
    for (let j = 0; j < 2 * n; j++) {
      const qpPlus = [...qp0];
      const qpMinus = [...qp0];
      qpPlus[j] = qpPlus[j]! + eps;
      qpMinus[j] = qpMinus[j]! - eps;

      const fPlus = leapfrogMap(qpPlus);
      const fMinus = leapfrogMap(qpMinus);

      const col: number[] = [];
      for (let i = 0; i < 2 * n; i++) {
        col.push((fPlus[i]! - fMinus[i]!) / (2 * eps));
      }
      jacobian.push(col);
    }

    // Transpose to get correct orientation (rows are output components)
    const jacobianT: number[][] = [];
    for (let i = 0; i < 2 * n; i++) {
      const row: number[] = [];
      for (let j = 0; j < 2 * n; j++) {
        row.push(jacobian[j]![i]!);
      }
      jacobianT.push(row);
    }
    return jacobianT;
  }

  /**
   * Compute determinant of a matrix using LU decomposition.
   */
  function determinant(matrix: number[][]): number {
    const n = matrix.length;
    // Copy matrix to avoid mutation
    const A = matrix.map((row) => [...row]);

    let det = 1;
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(A[k]![i]!) > Math.abs(A[maxRow]![i]!)) {
          maxRow = k;
        }
      }

      // Swap rows
      if (maxRow !== i) {
        [A[i], A[maxRow]] = [A[maxRow]!, A[i]!];
        det *= -1;
      }

      if (Math.abs(A[i]![i]!) < 1e-10) {
        return 0; // Singular matrix
      }

      det *= A[i]![i]!;

      // Eliminate column
      for (let k = i + 1; k < n; k++) {
        const factor = A[k]![i]! / A[i]![i]!;
        for (let j = i; j < n; j++) {
          A[k]![j]! -= factor * A[i]![j]!;
        }
      }
    }

    return det;
  }

  test("Jacobian determinant equals 1 (1D)", async () => {
    const q0 = [1.0];
    const p0 = [0.5];
    const stepSize = 0.1;
    const numSteps = 10;
    const tolerance = 1e-4;

    const jacobian = computeJacobian(q0, p0, stepSize, numSteps);
    const det = determinant(jacobian);

    expect(Math.abs(det - 1.0)).toBeLessThan(tolerance);
  });

  test("Jacobian determinant equals 1 (2D)", async () => {
    const q0 = [1.0, -0.5];
    const p0 = [0.3, 0.7];
    const stepSize = 0.1;
    const numSteps = 10;
    const tolerance = 1e-4;

    const jacobian = computeJacobian(q0, p0, stepSize, numSteps);
    const det = determinant(jacobian);

    expect(Math.abs(det - 1.0)).toBeLessThan(tolerance);
  });

  test("Jacobian determinant equals 1 (3D)", async () => {
    const q0 = [0.5, -0.3, 1.2];
    const p0 = [0.1, -0.4, 0.2];
    const stepSize = 0.1;
    const numSteps = 10;
    const tolerance = 1e-4;

    const jacobian = computeJacobian(q0, p0, stepSize, numSteps);
    const det = determinant(jacobian);

    expect(Math.abs(det - 1.0)).toBeLessThan(tolerance);
  });

  test("volume preservation with different step sizes", async () => {
    const q0 = [1.0, 0.5];
    const p0 = [0.2, -0.3];
    const numSteps = 10;
    const tolerance = 1e-4;

    for (const stepSize of [0.05, 0.1, 0.2]) {
      const jacobian = computeJacobian(q0, p0, stepSize, numSteps);
      const det = determinant(jacobian);
      expect(Math.abs(det - 1.0)).toBeLessThan(tolerance);
    }
  });

  test("volume preservation over many steps", async () => {
    const q0 = [0.8, -0.6];
    const p0 = [0.4, 0.5];
    const stepSize = 0.05;
    const numSteps = 50;
    const tolerance = 1e-4;

    const jacobian = computeJacobian(q0, p0, stepSize, numSteps);
    const det = determinant(jacobian);

    expect(Math.abs(det - 1.0)).toBeLessThan(tolerance);
  });
});
