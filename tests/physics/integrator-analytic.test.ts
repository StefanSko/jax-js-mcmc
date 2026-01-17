import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np, type Array } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

function runLeapfrog1D(
  q0: number,
  p0: number,
  gradLogProb: (q: Array) => Array,
  stepSize: number,
  numSteps: number,
): { q: number; p: number } {
  const q = np.array([q0]);
  const p = np.array([p0]);
  const massMatrix = np.onesLike(q.ref);
  const [q1, p1] = leapfrog(q.ref, p.ref, gradLogProb, stepSize, numSteps, massMatrix);
  return { q: q1.dataSync()[0], p: p1.dataSync()[0] };
}

describe("leapfrog analytic trajectory", () => {
  test("harmonic oscillator follows analytic solution", () => {
    const gradLogProb = (q: Array) => q.mul(-1);
    const stepSize = 0.01;
    const numSteps = 100;
    const t = stepSize * numSteps;

    const { q, p } = runLeapfrog1D(1.0, 0.0, gradLogProb, stepSize, numSteps);

    expect(Math.abs(q - Math.cos(t))).toBeLessThan(1e-2);
    expect(Math.abs(p - -Math.sin(t))).toBeLessThan(1e-2);
  });

  test("free fall matches quadratic trajectory", () => {
    const g = 1.0;
    const gradLogProb = (q: Array) => np.onesLike(q.ref).mul(-g);
    const stepSize = 0.01;
    const numSteps = 100;
    const t = stepSize * numSteps;

    const { q, p } = runLeapfrog1D(0.0, 1.0, gradLogProb, stepSize, numSteps);

    const qExpected = 0.0 + 1.0 * t - 0.5 * g * t * t;
    const pExpected = 1.0 - g * t;

    expect(Math.abs(q - qExpected)).toBeLessThan(1e-2);
    expect(Math.abs(p - pExpected)).toBeLessThan(1e-2);
  });
});
