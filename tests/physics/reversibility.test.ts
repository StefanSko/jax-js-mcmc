import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, grad, type Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog";

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
});
