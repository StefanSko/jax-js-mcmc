/**
 * ESS Diagnostics Tests
 */

import { describe, expect, test } from "vitest";

import { ess } from "../../src/diagnostics";

function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

describe("ESS diagnostics", () => {
  test("does not exceed chain length for iid draws", () => {
    const rng = lcg(123);
    const chain = Array.from({ length: 200 }, () => rng() - 0.5);

    const essValue = ess([chain]);

    expect(essValue).toBeLessThanOrEqual(chain.length + 1e-6);
    expect(essValue).toBeGreaterThan(chain.length * 0.9);
  });

  test("combines per-chain ESS without cross-chain autocorrelation", () => {
    const chain1 = [0, 1, 0, 1, 0, 1];
    const chain2 = [1, 0, 1, 0, 1, 0];

    const combinedEss = ess([chain1, chain2]);
    const perChainEss = ess([chain1]) + ess([chain2]);

    expect(combinedEss).toBeCloseTo(perChainEss, 10);
  });
});
