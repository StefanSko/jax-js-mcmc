import { numpy as np, random, grad, Array, type JsTree } from "@jax-js/jax";
import { leapfrog } from "./leapfrog";
import { kineticEnergy } from "./hamiltonian";
import {
  clampStepSize,
  findReasonableStepSize,
  initDualAverage,
  initMassMatrix,
  initMassMatrixState,
  updateDualAverage,
  updateMassMatrix,
  finalizeMassMatrix,
} from "./adaptation";
import { mapTree, stackTrees, treeClone, treeDispose, treeRef } from "./tree-utils";
import { splitKeys, sampleNormalTree } from "./random-utils";

export interface HMCOptions<Params extends JsTree<Array>> {
  numSamples: number;
  initialParams: Params;
  key: Array;

  numWarmup?: number;
  numLeapfrogSteps?: number;
  numChains?: number;

  initialStepSize?: number;
  targetAcceptRate?: number;

  adaptMassMatrix?: boolean;
}

export interface HMCStats<Params extends JsTree<Array>> {
  acceptRate: number;
  acceptRatePerChain: number[];
  stepSize: number;
  stepSizePerChain: number[];
  massMatrix: Params | Params[];
}

export interface HMCResult<Params extends JsTree<Array>> {
  draws: Params;
  stats: HMCStats<Params>;
}

function sampleMomentum<Params extends JsTree<Array>>(
  key: Array,
  massMatrix: Params,
): { nextKey: Array; momentum: Params } {
  const { nextKey, sample } = sampleNormalTree(key, massMatrix);
  const momentum = mapTree(
    (z: Array, m: Array) => z.mul(np.sqrt(m.ref)),
    sample,
    massMatrix,
  ) as Params;
  return { nextKey, momentum };
}

async function runChain<Params extends JsTree<Array>>(
  logProb: (p: Params) => Array,
  options: Required<HMCOptions<Params>>,
  chainKey: Array,
): Promise<{
  draws: Params;
  acceptRate: number;
  stepSize: number;
  massMatrix: Params;
}> {
  const {
    numSamples,
    numWarmup,
    numLeapfrogSteps,
    initialStepSize,
    targetAcceptRate,
    adaptMassMatrix,
  } = options;

  let key = chainKey;
  let position = treeClone(options.initialParams);
  const gradLogProb = grad(logProb) as (p: Params) => Params;

  let massMatrix = initMassMatrix(position);
  let massState = initMassMatrixState(position);

  if (numWarmup > 0) {
    const stepInit = findReasonableStepSize(
      logProb,
      treeClone(position),
      massMatrix,
      initialStepSize,
      key,
    );
    key = stepInit.nextKey;
    var stepSize = stepInit.stepSize;
    var dualState = initDualAverage(stepSize);
  } else {
    var stepSize = clampStepSize(initialStepSize);
    var dualState = initDualAverage(stepSize);
  }

  const samples: Params[] = [];
  let acceptCount = 0;

  const totalIters = numWarmup + numSamples;
  for (let iter = 0; iter < totalIters; iter++) {
    const keys = splitKeys(key, 3);
    key = keys[0];
    const momentumKey = keys[1];
    const acceptKey = keys[2];

    const { momentum } = sampleMomentum(momentumKey, massMatrix);

    const logProbCurrent = logProb(treeClone(position));
    const kineticCurrent = kineticEnergy(treeRef(momentum), massMatrix);

    const [proposalQ, proposalP] = leapfrog(
      treeClone(position),
      momentum,
      gradLogProb,
      stepSize,
      numLeapfrogSteps,
      massMatrix,
    );

    const logProbProposal = logProb(treeClone(proposalQ));
    const kineticProposal = kineticEnergy(proposalP, massMatrix);

    const logAccept = logProbProposal
      .sub(logProbCurrent)
      .sub(kineticProposal.sub(kineticCurrent))
      .item();
    const acceptProb = Math.min(1, Math.exp(logAccept));

    const u = random.uniform(acceptKey).item();
    const accepted = u < acceptProb;

    if (accepted) {
      treeDispose(position);
      position = proposalQ;
    } else {
      treeDispose(proposalQ);
    }

    if (iter < numWarmup) {
      dualState = updateDualAverage(dualState, acceptProb, targetAcceptRate);
      stepSize = clampStepSize(Math.exp(dualState.logStepSize));

      if (adaptMassMatrix) {
        massState = updateMassMatrix(massState, position);
      }

      if (iter === numWarmup - 1) {
        const tunedStepSize = clampStepSize(Math.exp(dualState.logStepSizeAvg));
        if (adaptMassMatrix) {
          massMatrix = finalizeMassMatrix(massState, massMatrix);
          const stepInit = findReasonableStepSize(
            logProb,
            treeClone(position),
            massMatrix,
            tunedStepSize,
            key,
          );
          key = stepInit.nextKey;
          stepSize = stepInit.stepSize;
        } else {
          stepSize = tunedStepSize;
        }
      }
    } else {
      if (accepted) acceptCount++;
      samples.push(treeClone(position));
    }

  }

  const acceptRate = numSamples > 0 ? acceptCount / numSamples : 0;
  const draws = stackTrees(samples, 0);

  treeDispose(position);
  return { draws, acceptRate, stepSize, massMatrix };
}

export async function hmc<Params extends JsTree<Array>>(
  logProb: (p: Params) => Array,
  options: HMCOptions<Params>,
): Promise<HMCResult<Params>> {
  if (options.numSamples <= 0) {
    throw new Error("numSamples must be > 0");
  }
  const opts: Required<HMCOptions<Params>> = {
    numSamples: options.numSamples,
    initialParams: options.initialParams,
    key: options.key,
    numWarmup: options.numWarmup ?? 1000,
    numLeapfrogSteps: options.numLeapfrogSteps ?? 25,
    numChains: options.numChains ?? 1,
    initialStepSize: options.initialStepSize ?? 0.1,
    targetAcceptRate: options.targetAcceptRate ?? 0.8,
    adaptMassMatrix: options.adaptMassMatrix ?? true,
  };

  const chainKeys = splitKeys(opts.key, opts.numChains);
  const chainResults = [] as {
    draws: Params;
    acceptRate: number;
    stepSize: number;
    massMatrix: Params;
  }[];

  for (let i = 0; i < opts.numChains; i++) {
    chainResults.push(await runChain(logProb, opts, chainKeys[i]));
  }

  const draws = stackTrees(
    chainResults.map((c) => c.draws),
    0,
  );

  const acceptRatePerChain = chainResults.map((c) => c.acceptRate);
  const stepSizePerChain = chainResults.map((c) => c.stepSize);
  const acceptRate =
    acceptRatePerChain.reduce((a, b) => a + b, 0) /
    (acceptRatePerChain.length || 1);
  const stepSize =
    stepSizePerChain.reduce((a, b) => a + b, 0) /
    (stepSizePerChain.length || 1);

  const stats: HMCStats<Params> = {
    acceptRate,
    acceptRatePerChain,
    stepSize,
    stepSizePerChain,
    massMatrix:
      opts.numChains === 1
        ? chainResults[0].massMatrix
        : chainResults.map((c) => c.massMatrix),
  };

  return { draws, stats };
}
