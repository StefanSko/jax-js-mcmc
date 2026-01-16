import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, random, Array, tree } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog.js";

/**
 * Detailed Balance Tests
 *
 * HMC satisfies detailed balance with Metropolis-Hastings correction.
 * The acceptance probability should follow: α = min(1, exp(-ΔH))
 *
 * We verify this by running many proposals and binning by energy difference,
 * then checking that the observed acceptance rate matches the theoretical prediction.
 *
 * Tolerance: ±0.1 on acceptance rate
 */

describe("HMC detailed balance", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  /**
   * Log probability for standard normal: logProb(q) = -0.5 * q^T * q
   */
  function logProb(q: Array): Array {
    return q.mul(q).mul(-0.5).sum();
  }

  /**
   * Gradient of log prob for standard normal: grad(logProb) = -q
   * Using analytical gradient to avoid tracer complexity in tests.
   */
  function gradLogProb(q: Array): Array {
    return q.mul(-1) as Array;
  }

  /**
   * Compute Hamiltonian: H(q, p) = U(q) + K(p) = -logProb(q) + 0.5 * p^T * p
   */
  function hamiltonian(q: Array, p: Array): number {
    const U = logProb(q).mul(-1);
    const K = p.mul(p).mul(0.5).sum();
    return (U.add(K).item() as number);
  }

  /**
   * Run a single HMC proposal and return energy difference and whether to accept.
   */
  function hmcProposal(
    q: Array,
    key: Array,
    stepSize: number,
    numSteps: number,
  ): { deltaH: number; accept: boolean; acceptProb: number } {
    // Sample momentum
    const p = random.normal(key, q.shape);

    // Initial Hamiltonian (use .ref for arrays that will be reused)
    const H0 = hamiltonian(q.ref, p.ref);

    // Leapfrog integration
    const [qNew, pNew] = leapfrog(q, p, gradLogProb as any, stepSize, numSteps);

    // Final Hamiltonian
    const H1 = hamiltonian(qNew as Array, pNew as Array);

    const deltaH = H1 - H0;
    const acceptProb = Math.min(1, Math.exp(-deltaH));
    const accept = Math.random() < acceptProb;

    return { deltaH, accept, acceptProb };
  }

  test("acceptance probability follows Metropolis rule", async () => {
    const stepSize = 0.1;
    const numSteps = 10;
    const numTrials = 500; // Reduced for faster tests
    const tolerance = 0.1;

    // Bins for energy difference
    const bins: Map<string, { count: number; accepted: number; totalDeltaH: number }> = new Map();

    function getBinKey(deltaH: number): string {
      // Bin by rounded energy difference
      if (deltaH < -1) return "< -1";
      if (deltaH < 0) return "[-1, 0)";
      if (deltaH < 0.5) return "[0, 0.5)";
      if (deltaH < 1) return "[0.5, 1)";
      return ">= 1";
    }

    let baseKey = random.key(42);

    for (let i = 0; i < numTrials; i++) {
      // Split key for this trial - Array is iterable
      const [nextKey, posKey, momKey] = random.split(baseKey, 3);
      baseKey = nextKey!;

      // Random starting position
      const q = random.normal(posKey!, [3]);

      const { deltaH, accept } = hmcProposal(q, momKey!, stepSize, numSteps);

      const binKey = getBinKey(deltaH);
      const bin = bins.get(binKey) || { count: 0, accepted: 0, totalDeltaH: 0 };
      bin.count++;
      if (accept) bin.accepted++;
      bin.totalDeltaH += deltaH;
      bins.set(binKey, bin);
    }

    // Check acceptance rates by bin
    for (const [binKey, bin] of bins) {
      if (bin.count < 10) continue; // Skip bins with too few samples

      const meanDeltaH = bin.totalDeltaH / bin.count;
      const expectedAcceptRate = Math.min(1, Math.exp(-meanDeltaH));
      const observedAcceptRate = bin.accepted / bin.count;

      // Allow tolerance of ±0.1
      expect(Math.abs(observedAcceptRate - expectedAcceptRate)).toBeLessThan(tolerance);
    }
  });

  test("perfect acceptance when step size is small", async () => {
    // With very small step size, energy is almost conserved, so acceptance should be ~1
    const stepSize = 0.01;
    const numSteps = 5;
    const numTrials = 50;

    let acceptCount = 0;
    let baseKey = random.key(123);

    for (let i = 0; i < numTrials; i++) {
      const [nextKey, posKey, momKey] = random.split(baseKey, 3);
      baseKey = nextKey!;

      const q = random.normal(posKey!, [2]);
      const { accept } = hmcProposal(q, momKey!, stepSize, numSteps);
      if (accept) acceptCount++;
    }

    const acceptRate = acceptCount / numTrials;
    expect(acceptRate).toBeGreaterThan(0.9); // Should be very high
  });

  test("lower acceptance when step size is large", async () => {
    // With large step size, energy drift increases, so acceptance should decrease
    const stepSize = 0.5;
    const numSteps = 20;
    const numTrials = 50;

    let acceptCount = 0;
    let baseKey = random.key(456);

    for (let i = 0; i < numTrials; i++) {
      const [nextKey, posKey, momKey] = random.split(baseKey, 3);
      baseKey = nextKey!;

      const q = random.normal(posKey!, [2]);
      const { accept } = hmcProposal(q, momKey!, stepSize, numSteps);
      if (accept) acceptCount++;
    }

    const acceptRate = acceptCount / numTrials;
    // With large step size, acceptance should be lower (but not zero)
    expect(acceptRate).toBeLessThan(0.95);
    expect(acceptRate).toBeGreaterThan(0.1);
  });
});
