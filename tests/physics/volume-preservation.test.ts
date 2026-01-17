import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, grad, type Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (q: Array) => q.ref.mul(q).mul(-0.5).sum();
const gradLogProb = grad(logProb) as (q: Array) => Array;

function computeJacobian(
  fn: (x: number[]) => number[],
  x0: number[],
  eps = 1e-4,
): number[][] {
  const n = x0.length;
  const jac: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const xPlus = x0.slice();
    const xMinus = x0.slice();
    xPlus[i] += eps;
    xMinus[i] -= eps;
    const fPlus = fn(xPlus);
    const fMinus = fn(xMinus);
    for (let j = 0; j < n; j++) {
      jac[j][i] = (fPlus[j] - fMinus[j]) / (2 * eps);
    }
  }
  return jac;
}

describe("leapfrog volume preservation", () => {
  test("jacobian determinant equals 1", () => {
    const q0 = np.array([1.0, -0.5]);
    const p0 = np.array([0.3, 0.9]);
    const massMatrix = np.onesLike(q0.ref);
    const stepSize = 0.1;
    const numSteps = 10;

    const fn = (x: number[]) => {
      const q = np.array([x[0], x[1]]);
      const p = np.array([x[2], x[3]]);
      const [q1, p1] = leapfrog(q, p, gradLogProb, stepSize, numSteps, massMatrix);
      return [...q1.dataSync(), ...p1.dataSync()];
    };

    const x0 = [...q0.dataSync(), ...p0.dataSync()];
    const jac = computeJacobian(fn, x0);
    const det = np.linalg.det(np.array(jac)).item();
    expect(Math.abs(det - 1.0)).toBeLessThan(3e-4);
  });
});
