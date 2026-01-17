import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, grad, type Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog";
import { treeMulScalar, treeOnesLike, treeRef } from "../../src/tree-utils";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (q: Array) => q.ref.mul(q).mul(-0.5).sum();
const gradLogProb = grad(logProb) as (q: Array) => Array;

describe("leapfrog reversibility", () => {
  test("forward then backward returns to start", () => {
    const q0 = np.array([1.0, 2.0, 3.0]);
    const p0 = np.array([0.5, -0.5, 0.1]);
    const massMatrix = np.onesLike(q0.ref);
    const stepSize = 0.1;
    const numSteps = 25;

    const [q1, p1] = leapfrog(q0.ref, p0.ref, gradLogProb, stepSize, numSteps, massMatrix);
    const [q2, p2] = leapfrog(q1, p1.mul(-1), gradLogProb, stepSize, numSteps, massMatrix);

    expect(np.allclose(q2, q0, { atol: 1e-5, rtol: 0 })).toBe(true);
    expect(np.allclose(p2.mul(-1), p0, { atol: 1e-5, rtol: 0 })).toBe(true);
  });

  test("single step reversibility", () => {
    const q0 = np.array([1.5, -2.0, 0.7]);
    const p0 = np.array([0.5, 0.3, -0.2]);
    const massMatrix = np.onesLike(q0.ref);

    const [q1, p1] = leapfrog(
      q0.ref,
      p0.ref,
      gradLogProb,
      0.2,
      1,
      massMatrix,
    );

    const [q2, p2Neg] = leapfrog(
      q1.ref,
      p1.mul(-1),
      gradLogProb,
      0.2,
      1,
      massMatrix,
    );
    const p2 = p2Neg.mul(-1);

    const qDiff = np.max(np.abs(q2.sub(q0))).item();
    const pDiff = np.max(np.abs(p2.sub(p0))).item();

    expect(qDiff).toBeLessThan(1e-5);
    expect(pDiff).toBeLessThan(1e-5);
  });

  test("reversibility with pytree parameters", () => {
    const gradTree = (params: { x: Array; y: Array }) => ({
      x: params.x.mul(-1),
      y: params.y.mul(-2),
    });

    const q0 = { x: np.array([1.0, -0.5]), y: np.array([2.0]) };
    const p0 = { x: np.array([0.3, 0.1]), y: np.array([-0.4]) };
    const massMatrix = treeOnesLike(treeRef(q0));

    const [q1, p1] = leapfrog(
      treeRef(q0),
      treeRef(p0),
      gradTree,
      0.15,
      15,
      massMatrix,
    );

    const p1Neg = treeMulScalar(treeRef(p1), -1);
    const [q2, p2Neg] = leapfrog(
      treeRef(q1),
      p1Neg,
      gradTree,
      0.15,
      15,
      massMatrix,
    );
    const p2 = treeMulScalar(p2Neg, -1);

    function maxDiff(a: Array, b: Array): number {
      return np.max(np.abs(a.sub(b))).item();
    }

    expect(maxDiff(q2.x, q0.x)).toBeLessThan(1e-5);
    expect(maxDiff(q2.y, q0.y)).toBeLessThan(1e-5);
    expect(maxDiff(p2.x, p0.x)).toBeLessThan(1e-5);
    expect(maxDiff(p2.y, p0.y)).toBeLessThan(1e-5);
  });

  test("reversibility with non-identity mass matrix", () => {
    const q0 = np.array([1.0, -0.5, 2.0]);
    const p0 = np.array([0.3, 0.1, -0.4]);
    const massMatrix = np.array([2.0, 0.5, 1.5]);

    const [q1, p1] = leapfrog(
      q0.ref,
      p0.ref,
      gradLogProb,
      0.1,
      20,
      massMatrix,
    );

    const [q2, p2Neg] = leapfrog(
      q1.ref,
      p1.mul(-1),
      gradLogProb,
      0.1,
      20,
      massMatrix,
    );
    const p2 = p2Neg.mul(-1);

    const qDiff = np.max(np.abs(q2.sub(q0))).item();
    const pDiff = np.max(np.abs(p2.sub(p0))).item();

    expect(qDiff).toBeLessThan(1e-5);
    expect(pDiff).toBeLessThan(1e-5);
  });

  test("reversibility with non-quadratic potential", () => {
    const logProbQuartic = (q: Array) => {
      const qSquared = q.ref.mul(q);
      return qSquared.ref.mul(qSquared).mul(-0.25).sum();
    };
    const gradQuartic = grad(logProbQuartic) as (q: Array) => Array;

    const q0 = np.array([0.8, -0.6]);
    const p0 = np.array([0.2, 0.3]);
    const massMatrix = np.onesLike(q0.ref);

    const [q1, p1] = leapfrog(
      q0.ref,
      p0.ref,
      gradQuartic,
      0.05,
      30,
      massMatrix,
    );

    const [q2, p2Neg] = leapfrog(
      q1.ref,
      p1.mul(-1),
      gradQuartic,
      0.05,
      30,
      massMatrix,
    );
    const p2 = p2Neg.mul(-1);

    const qDiff = np.max(np.abs(q2.sub(q0))).item();
    const pDiff = np.max(np.abs(p2.sub(p0))).item();

    expect(qDiff).toBeLessThan(1e-4);
    expect(pDiff).toBeLessThan(1e-4);
  });
});
