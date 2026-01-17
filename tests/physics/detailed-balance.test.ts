import { beforeAll, describe, expect, test } from "vitest";
import {
  defaultDevice,
  init,
  numpy as np,
  random,
  grad,
  type Array,
} from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog";
import { hamiltonian } from "../../src/hamiltonian";
import { splitKeys, sampleNormalTree } from "../../src/random-utils";
import { mapTree } from "../../src/tree-utils";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

const logProb = (q: Array) => q.ref.mul(q).mul(-0.5).sum();
const gradLogProb = grad(logProb) as (q: Array) => Array;

describe("HMC detailed balance", () => {
  test("acceptance probability follows Metropolis rule", () => {
    const numTrials = 5000;
    const stepSize = 0.25;
    const numSteps = 10;
    const q0Base = np.array([0.1]);
    const massMatrix = np.onesLike(q0Base.ref);

    let key = random.key(0);
    const binEdges = [0, 0.1, 0.2, 0.5, 1, 2, 4];
    const binCounts = new Array(binEdges.length).fill(0);
    const binAccepts = new Array(binEdges.length).fill(0);
    const binDeltaSum = new Array(binEdges.length).fill(0);

    for (let i = 0; i < numTrials; i++) {
      const keys = splitKeys(key, 3);
      key = keys[0];
      const momentumKey = keys[1];
      const acceptKey = keys[2];

      const { sample: z } = sampleNormalTree(momentumKey, massMatrix);
      const momentum = mapTree(
        (zi: Array, m: Array) => zi.mul(np.sqrt(m.ref)),
        z,
        massMatrix,
      ) as Array;

      const q0 = q0Base.ref;
      const h0 = hamiltonian(q0Base.ref, momentum, logProb, massMatrix);
      const [q1, p1] = leapfrog(
        q0,
        momentum,
        gradLogProb,
        stepSize,
        numSteps,
        massMatrix,
      );
      const h1 = hamiltonian(q1, p1, logProb, massMatrix);
      const deltaH = h1.sub(h0).item();
      const acceptProb = Math.min(1, Math.exp(-deltaH));
      const u = random.uniform(acceptKey).item();
      const accepted = u < acceptProb;

      const delta = Math.max(0, deltaH);
      let bin = binEdges.findIndex((edge, idx) =>
        idx === binEdges.length - 1
          ? delta >= edge
          : delta >= edge && delta < binEdges[idx + 1],
      );
      if (bin === -1) bin = binEdges.length - 1;

      binCounts[bin] += 1;
      binAccepts[bin] += accepted ? 1 : 0;
      binDeltaSum[bin] += delta;
    }

    for (let i = 0; i < binEdges.length; i++) {
      if (binCounts[i] < 50) continue;
      const meanDelta = binDeltaSum[i] / binCounts[i];
      const expected = Math.min(1, Math.exp(-meanDelta));
      const observed = binAccepts[i] / binCounts[i];
      expect(Math.abs(observed - expected)).toBeLessThan(0.1);
    }
  });
});
