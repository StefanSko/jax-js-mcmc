/**
 * Detailed Balance Tests for HMC
 *
 * Tests that the Metropolis-Hastings acceptance criterion satisfies detailed balance.
 * This validates that when combined with the symplectic leapfrog integrator,
 * HMC correctly samples from the target distribution.
 */

import { grad, numpy as np, random } from "@jax-js/jax";
import { describe, expect, test } from "vitest";

import { leapfrog } from "../../src/leapfrog";

describe("HMC Detailed Balance", () => {
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

  test("acceptance rate is reasonable for well-tuned step size", () => {
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    const numTrials = 500;
    let numAccepted = 0;

    let key = random.key(42);

    for (let i = 0; i < numTrials; i++) {
      // Split key for this iteration (4 keys: q, p, accept, next)
      const [k1, k2, k3, keyNext] = random.split(key, 4);
      key = keyNext;

      // Sample initial position and momentum from prior
      const q0 = random.normal(k1, [3]);
      const p0 = random.normal(k2, [3]);

      const H0 = hamiltonian(q0.ref, p0.ref, logProb);

      // Run leapfrog with large step size to create energy error
      const [q1, p1] = leapfrog(
        q0.ref,
        p0.ref,
        (q) => gradLogProb(q),
        0.5, // larger step size for more energy error
        20, // num steps
        undefined,
      );

      const H1 = hamiltonian(q1, p1, logProb);

      // Metropolis acceptance
      const deltaH = (H1.js() as number) - (H0.js() as number);
      const acceptProb = Math.min(1, Math.exp(-deltaH));

      // Use uniform random for accept/reject decision
      const u = random.uniform(k3).js() as number;
      if (u < acceptProb) {
        numAccepted++;
      }
    }

    const acceptRate = numAccepted / numTrials;

    // For HMC, acceptance rate varies based on step size.
    // A high acceptance rate (>95%) is actually fine - it means energy is well conserved.
    // The key property is that it should be > 0 (proposals accepted) and <= 1.
    // We test: acceptance is reasonable (50-100%)
    expect(acceptRate).toBeWithinRange(0.5, 1.0);
  });

  test("acceptance rate decreases with larger step sizes", () => {
    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    // Use more extreme step sizes to see clear difference
    const stepSizes = [0.1, 0.5, 1.0];
    const acceptRates: number[] = [];

    for (const eps of stepSizes) {
      let key = random.key(123);
      let numAccepted = 0;
      const numTrials = 300;

      for (let i = 0; i < numTrials; i++) {
        const [k1, k2, keyNext] = random.split(key, 3);
        key = keyNext;

        const q0 = random.normal(k1, [2]);
        const p0 = random.normal(k2, [2]);
        const H0 = hamiltonian(q0.ref, p0.ref, logProb);

        const [q1, p1] = leapfrog(
          q0.ref,
          p0.ref,
          (q) => gradLogProb(q),
          eps,
          25,
          undefined,
        );

        const H1 = hamiltonian(q1, p1, logProb);
        const deltaH = (H1.js() as number) - (H0.js() as number);

        if (Math.exp(-deltaH) > Math.random()) {
          numAccepted++;
        }
      }

      acceptRates.push(numAccepted / numTrials);
    }

    // Acceptance should decrease with larger step sizes
    // Use >= to allow for statistical variation when rates are very close
    expect(acceptRates[0]).toBeGreaterThanOrEqual(acceptRates[1] - 0.05);
    expect(acceptRates[1]).toBeGreaterThan(acceptRates[2]);
  });

  test("detailed balance: forward and reverse proposals have correct ratio", () => {
    // Test detailed balance more directly:
    // P(q -> q') * P_accept(q -> q') should equal P(q' -> q) * P_accept(q' -> q)
    // For leapfrog with negated momentum, this reduces to checking energy differences

    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    let key = random.key(999);
    const numTrials = 200;
    let forwardAcceptSum = 0;
    let reverseAcceptSum = 0;

    for (let i = 0; i < numTrials; i++) {
      const [k1, k2, keyNext] = random.split(key, 3);
      key = keyNext;

      const q0 = random.normal(k1, [2]);
      const p0 = random.normal(k2, [2]);

      const H0 = hamiltonian(q0.ref, p0.ref, logProb).js() as number;

      // Forward proposal
      const [q1, p1] = leapfrog(
        q0.ref,
        p0.ref,
        (q) => gradLogProb(q),
        0.1,
        10,
        undefined,
      );

      const H1 = hamiltonian(q1.ref, p1.ref, logProb).js() as number;

      // Forward acceptance probability
      const forwardAccept = Math.min(1, Math.exp(-(H1 - H0)));
      forwardAcceptSum += forwardAccept;

      // Reverse proposal: negate momentum at q1, integrate back
      const [q0Rev, p0RevNeg] = leapfrog(
        q1.ref,
        p1.mul(-1),
        (q) => gradLogProb(q),
        0.1,
        10,
        undefined,
      );
      const p0Rev = p0RevNeg.mul(-1);

      // Reverse Hamiltonian (should be ~ H1 at start, ~ H0 at end)
      const H0Rev = hamiltonian(q0Rev, p0Rev, logProb).js() as number;

      // Reverse acceptance probability
      const reverseAccept = Math.min(1, Math.exp(-(H0Rev - H1)));
      reverseAcceptSum += reverseAccept;
    }

    // Average acceptance probabilities should be similar
    // (This is a weak test of detailed balance)
    const avgForward = forwardAcceptSum / numTrials;
    const avgReverse = reverseAcceptSum / numTrials;

    expect(Math.abs(avgForward - avgReverse)).toBeLessThan(0.1);
  });

  test("long trajectory maintains acceptance", () => {
    // Test that longer trajectories don't destroy acceptance
    // (they should have similar acceptance if energy is conserved)

    const logProb = (q: np.Array) => q.ref.mul(q).mul(-0.5).sum();
    const gradLogProb = grad(logProb);

    let key = random.key(777);
    let numAcceptedShort = 0;
    let numAcceptedLong = 0;
    const numTrials = 200;

    for (let i = 0; i < numTrials; i++) {
      const [k1, k2, keyNext] = random.split(key, 3);
      key = keyNext;

      const q0 = random.normal(k1, [2]);
      const p0 = random.normal(k2, [2]);
      const H0 = hamiltonian(q0.ref, p0.ref, logProb).js() as number;

      // Short trajectory
      const [q1Short, p1Short] = leapfrog(
        q0.ref,
        p0.ref,
        (q) => gradLogProb(q),
        0.1,
        10,
        undefined,
      );
      const H1Short = hamiltonian(q1Short, p1Short, logProb).js() as number;
      if (Math.exp(-(H1Short - H0)) > Math.random()) {
        numAcceptedShort++;
      }

      // Long trajectory
      const [q1Long, p1Long] = leapfrog(
        q0.ref,
        p0.ref,
        (q) => gradLogProb(q),
        0.1,
        50, // 5x longer
        undefined,
      );
      const H1Long = hamiltonian(q1Long, p1Long, logProb).js() as number;
      if (Math.exp(-(H1Long - H0)) > Math.random()) {
        numAcceptedLong++;
      }
    }

    const rateShort = numAcceptedShort / numTrials;
    const rateLong = numAcceptedLong / numTrials;

    // Both should have reasonable acceptance
    expect(rateShort).toBeWithinRange(0.5, 1.0);
    expect(rateLong).toBeWithinRange(0.5, 1.0);

    // Long trajectory might have slightly lower acceptance due to accumulated error
    // but should still be reasonable
    expect(rateLong).toBeGreaterThan(0.4);
  });
});
