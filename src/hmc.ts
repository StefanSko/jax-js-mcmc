/**
 * HMC (Hamiltonian Monte Carlo) Sampler
 *
 * Implements HMC sampling with automatic step size adaptation.
 */

import { grad, numpy as np, random, tree } from "@jax-js/jax";
import type { JsTree, LogProbFn } from "./types";
import { leapfrog } from "./leapfrog";

/**
 * HMC sampler options
 */
export interface HMCOptions<T extends JsTree<np.Array>> {
  // Required
  initialParams: T;
  key: np.Array; // PRNGKey
  numSamples: number;

  // Optional with defaults
  numWarmup?: number; // Default: 1000
  numLeapfrogSteps?: number; // Default: 25
  numChains?: number; // Default: 1
  initialStepSize?: number; // Default: 0.1
  targetAcceptRate?: number; // Default: 0.8
  adaptMassMatrix?: boolean; // Default: true
}

/**
 * HMC sampler result
 */
export interface HMCResult<T extends JsTree<np.Array>> {
  draws: np.Array; // Shape: [numChains, numSamples, ...paramShape]
  stats: {
    acceptRate: np.Array; // Per-chain + mean
    stepSize: np.Array; // Per-chain final step sizes
    massMatrix?: T; // Per-chain diagonal mass matrices
    rhat?: number[]; // R-hat per parameter
    ess?: number[]; // ESS per parameter
  };
}

/**
 * Extract individual PRNG keys from a stacked key array.
 */
function extractKeys(keysArray: np.Array, count: number): np.Array[] {
  const keys: np.Array[] = [];
  for (let i = 0; i < count; i++) {
    keys.push(keysArray.ref.slice([i, i + 1]).reshape([2]));
  }
  return keys;
}

/**
 * Run HMC sampling on a log probability function.
 *
 * @param logProb The log probability function (must return scalar Array)
 * @param options HMC options
 * @returns Promise resolving to HMCResult with draws and statistics
 */
export async function hmc<T extends JsTree<np.Array>>(
  logProb: LogProbFn<T>,
  options: HMCOptions<T>,
): Promise<HMCResult<T>> {
  // Extract options with defaults
  const {
    initialParams,
    key,
    numSamples,
    numWarmup = 1000,
    numLeapfrogSteps = 25,
    numChains = 1,
    initialStepSize = 0.1,
    targetAcceptRate = 0.8,
    adaptMassMatrix = true,
  } = options;

  // Get gradient function
  const gradLogProb = grad(logProb);

  // Split key for chains
  const chainKeys = random.split(key, numChains);
  const keyList = extractKeys(chainKeys, numChains);

  // Run each chain
  const chainResults: { draws: np.Array; acceptRate: number; stepSize: number }[] = [];

  for (let chainIdx = 0; chainIdx < numChains; chainIdx++) {
    const chainKey = keyList[chainIdx];
    const result = await runSingleChain(
      logProb,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      gradLogProb as any,
      tree.ref(initialParams) as T,
      chainKey,
      numSamples,
      numWarmup,
      numLeapfrogSteps,
      initialStepSize,
      targetAcceptRate,
      adaptMassMatrix,
    );
    chainResults.push(result);
  }

  // Stack draws from all chains
  const drawsArrays = chainResults.map((r) => r.draws);
  const draws = np.stack(drawsArrays);

  // Compute stats
  const acceptRates = np.array(chainResults.map((r) => r.acceptRate));
  const stepSizes = np.array(chainResults.map((r) => r.stepSize));

  return {
    draws,
    stats: {
      acceptRate: acceptRates,
      stepSize: stepSizes,
    },
  };
}

/**
 * Run a single HMC chain
 */
async function runSingleChain<T extends JsTree<np.Array>>(
  logProb: LogProbFn<T>,
  gradLogProb: (params: T) => T,
  initialParams: T,
  key: np.Array,
  numSamples: number,
  numWarmup: number,
  numLeapfrogSteps: number,
  initialStepSize: number,
  targetAcceptRate: number,
  adaptMassMatrix: boolean,
): Promise<{ draws: np.Array; acceptRate: number; stepSize: number }> {
  let currentParams = initialParams;
  let stepSize = initialStepSize;

  // Dual averaging state for step size adaptation
  // Following Hoffman & Gelman "The No-U-Turn Sampler"
  let logStepSize = Math.log(stepSize);
  let logStepSizeAvg = Math.log(stepSize);
  let hSum = 0;
  const gamma = 0.05;
  const t0 = 10;
  const kappa = 0.75;
  // mu is the target that log step size regresses toward - use a more conservative target
  const mu = Math.log(initialStepSize); // Target is initial step size, not 10x

  // Mass matrix (identity for now, can be adapted)
  let massMatrix: T | undefined = undefined;

  // Welford state for mass matrix adaptation
  let welfordCount = 0;
  let welfordMean: T | undefined = undefined;
  let welfordM2: T | undefined = undefined;

  const samples: np.Array[] = [];
  let numAccepted = 0;
  const totalIterations = numWarmup + numSamples;

  for (let iter = 0; iter < totalIterations; iter++) {
    const isWarmup = iter < numWarmup;

    // Split key for this iteration
    const [momentumKey, acceptKey, keyNext] = random.split(key, 3);
    key = keyNext;

    // Sample momentum from N(0, massMatrix) or N(0, I)
    const momentum = sampleMomentum(currentParams, momentumKey, massMatrix);

    // Compute current Hamiltonian
    // Note: hamiltonian consumes position, so pass tree.ref
    // Note: We need momentum to survive for leapfrog, so pass tree.ref
    const currentH = hamiltonian(
      tree.ref(currentParams) as T,
      tree.ref(momentum) as T,
      logProb,
      massMatrix,
    );

    // Run leapfrog integration
    // Note: currentParams survives the iteration, pass tree.ref
    // Note: momentum is consumed by leapfrog (not needed after)
    const [proposedParams, proposedMomentum] = leapfrog(
      tree.ref(currentParams) as T,
      tree.ref(momentum) as T, // Need tree.ref because hamiltonian consumed the arrays inside
      gradLogProb,
      stepSize,
      numLeapfrogSteps,
      massMatrix,
    );

    // Compute proposed Hamiltonian
    // CRITICAL: proposedParams may be accepted, so we need to preserve it
    // Use tree.ref twice - once for hamiltonian, once to keep it alive
    const proposedH = hamiltonian(
      tree.ref(proposedParams) as T,
      proposedMomentum,
      logProb,
      massMatrix,
    );

    // Metropolis acceptance
    const currentHVal = currentH.js() as number;
    const proposedHVal = proposedH.js() as number;
    const deltaH = proposedHVal - currentHVal;
    const acceptProb = Math.min(1.0, Math.exp(-deltaH));

    const u = random.uniform(acceptKey).js() as number;
    const accepted = u < acceptProb;

    if (accepted) {
      // proposedParams is still valid because we used tree.ref above
      currentParams = tree.ref(proposedParams) as T;
      if (!isWarmup) {
        numAccepted++;
      }
    }

    // Step size adaptation during warmup
    if (isWarmup) {
      const iteration = iter + 1;
      // Handle NaN/Inf proposals by treating them as rejections (accept prob = 0)
      const safeAcceptProb = isFinite(acceptProb) ? acceptProb : 0;
      hSum += targetAcceptRate - safeAcceptProb;
      logStepSize = mu - (Math.sqrt(iteration) / gamma) * (hSum / (iteration + t0));
      const eta = Math.pow(iteration, -kappa);
      logStepSizeAvg = eta * logStepSize + (1 - eta) * logStepSizeAvg;
      stepSize = Math.exp(logStepSize);

      // Clamp step size - use more conservative upper bound during warmup
      stepSize = Math.max(1e-4, Math.min(0.5, stepSize));

      // Mass matrix adaptation using Welford's algorithm
      // CRITICAL: Use .ref inside callbacks to preserve currentParams for next iteration
      if (adaptMassMatrix) {
        welfordCount++;
        if (welfordMean === undefined) {
          welfordMean = tree.map(
            (leaf: np.Array) => leaf.ref.mul(1),
            tree.ref(currentParams) as JsTree<np.Array>,
          ) as T;
          welfordM2 = tree.map(
            (leaf: np.Array) => np.zeros(leaf.shape),
            tree.ref(currentParams) as JsTree<np.Array>,
          ) as T;
        } else {
          // Update Welford
          // Use sample.ref to preserve currentParams arrays
          const delta = tree.map(
            (sample: np.Array, mean: np.Array) => sample.ref.sub(mean.ref),
            tree.ref(currentParams) as JsTree<np.Array>,
            tree.ref(welfordMean) as JsTree<np.Array>,
          ) as T;

          welfordMean = tree.map(
            (mean: np.Array, d: np.Array) => mean.add(d.div(welfordCount)),
            welfordMean as JsTree<np.Array>,
            tree.ref(delta) as JsTree<np.Array>,
          ) as T;

          // Use sample.ref to preserve currentParams arrays
          const delta2 = tree.map(
            (sample: np.Array, mean: np.Array) => sample.ref.sub(mean.ref),
            tree.ref(currentParams) as JsTree<np.Array>,
            tree.ref(welfordMean) as JsTree<np.Array>,
          ) as T;

          welfordM2 = tree.map(
            (m2: np.Array, d: np.Array, d2: np.Array) => m2.add(d.ref.mul(d2)),
            welfordM2 as JsTree<np.Array>,
            delta as JsTree<np.Array>,
            delta2 as JsTree<np.Array>,
          ) as T;
        }
      }

      // Finalize step size and mass matrix at end of warmup
      if (iter === numWarmup - 1) {
        stepSize = Math.exp(logStepSizeAvg);
        stepSize = Math.max(1e-4, Math.min(0.5, stepSize));

        if (adaptMassMatrix && welfordCount > 1 && welfordM2 !== undefined) {
          massMatrix = tree.map(
            (m2: np.Array) => m2.div(welfordCount - 1).add(1e-5),
            welfordM2 as JsTree<np.Array>,
          ) as T;
        }
      }
    }

    // Collect samples after warmup
    if (!isWarmup) {
      // Flatten current params to array
      const flatParams = flattenTree(tree.ref(currentParams) as T);
      samples.push(flatParams);
    }
  }

  // Stack samples: [numSamples, paramDim]
  const draws = np.stack(samples);
  const acceptRate = numAccepted / numSamples;

  return { draws, acceptRate, stepSize };
}

/**
 * Sample momentum from N(0, massMatrix) or N(0, I)
 */
function sampleMomentum<T extends JsTree<np.Array>>(
  params: T,
  key: np.Array,
  massMatrix?: T,
): T {
  const [leaves] = tree.flatten(tree.ref(params) as JsTree<np.Array>);
  const numLeaves = (leaves as np.Array[]).length;
  const keysArray = random.split(key, numLeaves);
  const extractedKeys = extractKeys(keysArray, numLeaves);

  let keyIdx = 0;
  if (massMatrix === undefined) {
    return tree.map(
      (leaf: np.Array) => random.normal(extractedKeys[keyIdx++], leaf.shape),
      tree.ref(params) as JsTree<np.Array>,
    ) as T;
  }

  // Sample from N(0, diag(massMatrix)): p ~ sqrt(M) * z where z ~ N(0, I)
  return tree.map(
    (leaf: np.Array, m: np.Array) =>
      random.normal(extractedKeys[keyIdx++], leaf.shape).mul(np.sqrt(m)),
    tree.ref(params) as JsTree<np.Array>,
    tree.ref(massMatrix) as JsTree<np.Array>,
  ) as T;
}

/**
 * Compute Hamiltonian H = U(q) + K(p)
 */
function hamiltonian<T extends JsTree<np.Array>>(
  position: T,
  momentum: T,
  logProb: LogProbFn<T>,
  massMatrix?: T,
): np.Array {
  // U(q) = -logProb(q)
  const U = logProb(position).mul(-1);

  // K(p) = 0.5 * sum(p^2 / M) or 0.5 * sum(p^2)
  const K = computeKineticEnergy(momentum, massMatrix);

  return U.add(K);
}

/**
 * Compute kinetic energy K = 0.5 * sum(p^2 / M)
 * If massMatrix is undefined, uses identity (K = 0.5 * sum(p^2))
 */
function computeKineticEnergy<T extends JsTree<np.Array>>(
  momentum: T,
  massMatrix?: T,
): np.Array {
  const [momentumLeaves] = tree.flatten(tree.ref(momentum) as JsTree<np.Array>);
  const pLeaves = momentumLeaves as np.Array[];

  if (massMatrix === undefined) {
    let total = np.array(0);
    for (const p of pLeaves) {
      total = total.add(p.ref.mul(p).sum());
    }
    return total.mul(0.5);
  }

  const [massLeaves] = tree.flatten(tree.ref(massMatrix) as JsTree<np.Array>);
  const mLeaves = massLeaves as np.Array[];
  let total = np.array(0);
  for (let i = 0; i < pLeaves.length; i++) {
    const p = pLeaves[i];
    const m = mLeaves[i];
    total = total.add(p.ref.mul(p).div(m).sum());
  }
  return total.mul(0.5);
}

/**
 * Flatten a JsTree<Array> to a single 1D Array
 */
function flattenTree<T extends JsTree<np.Array>>(params: T): np.Array {
  const [leaves] = tree.flatten(params as JsTree<np.Array>);
  const arrLeaves = leaves as np.Array[];
  const flatLeaves = arrLeaves.map((leaf) => leaf.reshape([-1]));
  return np.concatenate(flatLeaves);
}
