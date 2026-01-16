import { describe, expect, test } from "vitest";

import { numpy as np, random, type Array as JaxArray } from "@jax-js/jax";

import { hmc } from "../../src/hmc";

function normalLogPdf(x: JaxArray, mean: JaxArray, scale: JaxArray): JaxArray {
  const z = np.divide(np.subtract(x, mean), scale.ref);
  const term = np.multiply(-0.5, np.square(z));
  return np.subtract(term, np.log(scale));
}

describe("windowed warmup on regression", () => {
  test("recovers known coefficient", async () => {
    const key = random.key(0);
    const [xKey, noiseKey] = np.split(random.split(key, 2), 2, 0).map((k) => np.squeeze(k));
    const x = random.normal(xKey, [30, 1]);
    const noise = random.normal(noiseKey, [30, 1]);
    const y = np.add(np.multiply(3, x.ref), noise);

    const logProb = (params: { coefs: JaxArray; logScale: JaxArray }) => {
      const coefsForMean = params.coefs.ref;
      const coefsForPrior = params.coefs.ref;
      const logScaleForExp = params.logScale.ref;
      const logScaleForPrior = params.logScale.ref;

      const scale = np.exp(logScaleForExp);
      const mean = np.multiply(x.ref, coefsForMean);
      const logLik = np.sum(normalLogPdf(y.ref, mean, scale));
      const prior = np.add(
        np.multiply(-0.5, np.square(coefsForPrior)),
        np.multiply(-0.5, np.square(logScaleForPrior)),
      );
      return np.add(logLik, prior);
    };

    const result = await hmc(logProb, {
      numSamples: 150,
      numWarmup: 80,
      numLeapfrogSteps: 8,
      initialParams: {
        coefs: np.array(1.0),
        logScale: np.array(0.0),
      },
      key: random.key(1),
    });

    const meanCoef = np.mean(result.draws.coefs).item() as number;
    expect(meanCoef).toBeCloseTo(3.0, 0);
  }, 10000);
});
