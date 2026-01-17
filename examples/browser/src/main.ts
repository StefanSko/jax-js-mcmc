import { hmc } from "jax-js-mcmc";
import { init, defaultDevice, numpy as np, random, type Array } from "@jax-js/jax";

const output = document.querySelector<HTMLPreElement>("#output");

function log(msg: string) {
  if (output) {
    output.textContent = msg;
  }
}

async function run() {
  await init();
  try {
    defaultDevice("webgpu");
  } catch {
    defaultDevice("cpu");
  }

  const logProb = (params: { x: Array }) =>
    params.x.ref.mul(params.x).mul(-0.5).sum();

  const result = await hmc(logProb, {
    numSamples: 500,
    numWarmup: 500,
    numLeapfrogSteps: 25,
    initialParams: { x: np.zeros([2]) },
    key: random.key(0),
  });

  const draws = await result.draws.x.jsAsync();
  log(
    [
      `accept rate: ${result.stats.acceptRate.toFixed(3)}`,
      `first 5 draws: ${JSON.stringify(draws.slice(0, 5))}`,
    ].join("\n"),
  );
}

run().catch((err) => {
  log(`error: ${String(err)}`);
  console.error(err);
});
