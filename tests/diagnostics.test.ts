import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { rhat, ess, summary } from "../src/diagnostics";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

describe("diagnostics", () => {
  test("rhat is ~1 for identical chains", () => {
    const draws = np.array([
      [0, 1, 0, 1],
      [0, 1, 0, 1],
    ]);
    const r = rhat(draws) as number;
    expect(Number.isFinite(r)).toBe(true);
    expect(r).toBeLessThan(1.1);
  });

  test("ess is finite", () => {
    const draws = np.array([
      [0, 1, 2, 3, 4, 5],
      [0, 1, 2, 3, 4, 5],
    ]);
    const e = ess(draws) as number;
    expect(Number.isFinite(e)).toBe(true);
  });

  test("summary returns expected mean", () => {
    const draws = np.array([
      [0, 1, 2, 3],
      [0, 1, 2, 3],
    ]);
    const stats = summary({ x: draws });
    expect(Math.abs(stats.x.mean - 1.5)).toBeLessThanOrEqual(1e-6);
  });
});
