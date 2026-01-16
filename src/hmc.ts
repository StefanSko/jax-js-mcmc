import {
  numpy as np,
  random,
  tree,
  valueAndGrad,
  type Array as JaxArray,
} from "@jax-js/jax";

import {
  buildWarmupSchedule,
  clampStepSize,
  finalizeMassMatrix,
  findReasonableStepSize,
  initDualAveraging,
  initMassMatrix,
  updateDualAveraging,
  updateMassMatrix,
  type AcceptProbEvaluator,
} from "./adaptation";
import { kineticEnergy, potentialEnergy } from "./hamiltonian";
import { leapfrog } from "./leapfrog";
import { acceptanceProbability } from "./metropolis";
import { splitKey } from "./random";
import {
  stackTrees,
  treeNegate,
  treeOnesLike,
  treeReciprocal,
  type PyTree,
} from "./tree";

export type LogProbFn<T extends PyTree> = (params: T) => JaxArray;

export type HMCState<T extends PyTree> = {
  position: T;
  logProb: JaxArray;
  gradLogProb: T;
};

export type HMCInfo = {
  acceptProb: number;
  accepted: boolean;
};

export type HMCOptions<T extends PyTree> = {
  numSamples: number;
  initialParams: T;
  key: JaxArray;
  numWarmup?: number;
  numLeapfrogSteps?: number;
  numChains?: number;
  initialStepSize?: number;
  targetAcceptRate?: number;
  adaptMassMatrix?: boolean;
  inverseMassMatrix?: T;
};

export type HMCResult<T extends PyTree> = {
  draws: T;
  stats: {
    acceptRate: number;
    stepSize: number;
    massMatrix: T;
  };
};

export function initState<T extends PyTree>(
  position: T,
  logProbGrad: (params: T) => [JaxArray, T],
): HMCState<T> {
  const [logProb, gradLogProb] = logProbGrad(tree.ref(position) as T);
  return { position, logProb, gradLogProb };
}

function sampleMomentum<T extends PyTree>(
  key: JaxArray,
  reference: T,
  inverseMassMatrix: T,
): T {
  const [leaves, treedef] = tree.flatten(tree.ref(reference) as T);
  const invLeaves = tree.leaves(tree.ref(inverseMassMatrix) as T);
  if (leaves.length !== invLeaves.length) {
    throw new Error("inverseMassMatrix must match parameter tree structure");
  }
  const keys = splitKey(key, leaves.length);
  const samples = leaves.map((leaf, index) => {
    const shape = np.shape(leaf) as number[];
    const sample = random.normal(keys[index], shape.length === 0 ? undefined : shape);
    const scale = np.sqrt(np.reciprocal(invLeaves[index]));
    return np.multiply(sample, scale);
  });
  return tree.unflatten(treedef, samples) as T;
}

export function hmcStep<T extends PyTree>(
  key: JaxArray,
  state: HMCState<T>,
  stepSize: number,
  inverseMassMatrix: T,
  numLeapfrogSteps: number,
  logProbGrad: (params: T) => [JaxArray, T],
): { state: HMCState<T>; info: HMCInfo } {
  const [momentumKey, acceptKey] = splitKey(key, 2);
  const momentum = sampleMomentum(momentumKey, state.position, inverseMassMatrix);

  const currentH = np.add(
    potentialEnergy(state.logProb.ref),
    kineticEnergy(momentum, inverseMassMatrix),
  );

  const gradPotential = (q: T) => treeNegate(logProbGrad(tree.ref(q) as T)[1]);
  const [proposalQ, proposalP] = leapfrog(
    state.position,
    momentum,
    gradPotential,
    stepSize,
    numLeapfrogSteps,
    inverseMassMatrix,
  );
  const flippedP = treeNegate(proposalP);
  const [proposalLogProb, proposalGrad] = logProbGrad(tree.ref(proposalQ) as T);

  const proposalH = np.add(
    potentialEnergy(proposalLogProb.ref),
    kineticEnergy(flippedP, inverseMassMatrix),
  );

  const delta = np.subtract(proposalH, currentH).item() as number;
  const acceptProb = acceptanceProbability(delta);
  const u = random.uniform(acceptKey).item() as number;
  const accepted = u < acceptProb;

  if (accepted) {
    return {
      state: {
        position: proposalQ,
        logProb: proposalLogProb,
        gradLogProb: proposalGrad,
      },
      info: { acceptProb, accepted },
    };
  }

  return {
    state,
    info: { acceptProb, accepted },
  };
}

function runWarmup<T extends PyTree>(
  key: JaxArray,
  state: HMCState<T>,
  logProbGrad: (params: T) => [JaxArray, T],
  numWarmup: number,
  numLeapfrogSteps: number,
  targetAcceptRate: number,
  initialStepSize: number,
  inverseMassMatrix: T,
  adaptMassMatrix: boolean,
): {
  key: JaxArray;
  state: HMCState<T>;
  stepSize: number;
  inverseMassMatrix: T;
} {
  const schedule = buildWarmupSchedule(numWarmup);
  let stepSize = initialStepSize;
  let currentKey = key;

  const evaluator: AcceptProbEvaluator = (evalKey, evalStepSize) => {
    const [stepKey, nextKey] = splitKey(evalKey, 2);
    const { info } = hmcStep(
      stepKey,
      state,
      evalStepSize,
      inverseMassMatrix,
      numLeapfrogSteps,
      logProbGrad,
    );
    return { acceptProb: info.acceptProb, nextKey };
  };

  const stepSearch = findReasonableStepSize(
    currentKey,
    evaluator,
    targetAcceptRate,
    initialStepSize,
  );
  stepSize = stepSearch.stepSize;
  currentKey = stepSearch.key;

  let dualState = initDualAveraging(stepSize);
  let invMass = inverseMassMatrix;
  let massState = initMassMatrix(state.position);

  for (const window of schedule) {
    for (let i = 0; i < window.size; i += 1) {
      const [stepKey, nextKey] = splitKey(currentKey, 2);
      currentKey = nextKey;
      const stepResult = hmcStep(
        stepKey,
        state,
        stepSize,
        invMass,
        numLeapfrogSteps,
        logProbGrad,
      );
      state = stepResult.state;

      dualState = updateDualAveraging(
        dualState,
        stepResult.info.acceptProb,
        targetAcceptRate,
      );
      stepSize = clampStepSize(Math.exp(dualState.logStep));

      if (adaptMassMatrix && window.adaptMassMatrix) {
        massState = updateMassMatrix(massState, state.position);
      }
    }

    if (adaptMassMatrix && window.adaptMassMatrix) {
      invMass = finalizeMassMatrix(massState);
      massState = initMassMatrix(state.position);
    }
  }

  const finalStepSize = clampStepSize(Math.exp(dualState.logStepAvg));

  return {
    key: currentKey,
    state,
    stepSize: finalStepSize,
    inverseMassMatrix: invMass,
  };
}

function runChain<T extends PyTree>(
  key: JaxArray,
  initialPosition: T,
  logProbGrad: (params: T) => [JaxArray, T],
  options: {
    numSamples: number;
    numWarmup: number;
    numLeapfrogSteps: number;
    targetAcceptRate: number;
    initialStepSize: number;
    inverseMassMatrix: T;
    adaptMassMatrix: boolean;
  },
): {
  draws: T;
  acceptRate: number;
  stepSize: number;
  inverseMassMatrix: T;
} {
  let state = initState(tree.ref(initialPosition) as T, logProbGrad);
  let stepSize = options.initialStepSize;
  let invMass = tree.ref(options.inverseMassMatrix) as T;
  let currentKey = key;

  if (options.numWarmup > 0) {
    const warmupResult = runWarmup(
      currentKey,
      state,
      logProbGrad,
      options.numWarmup,
      options.numLeapfrogSteps,
      options.targetAcceptRate,
      options.initialStepSize,
      invMass,
      options.adaptMassMatrix,
    );
    currentKey = warmupResult.key;
    state = warmupResult.state;
    stepSize = warmupResult.stepSize;
    invMass = warmupResult.inverseMassMatrix;
  }

  const samples: T[] = [];
  let acceptSum = 0;

  for (let i = 0; i < options.numSamples; i += 1) {
    const [stepKey, nextKey] = splitKey(currentKey, 2);
    currentKey = nextKey;
    const stepResult = hmcStep(
      stepKey,
      state,
      stepSize,
      invMass,
      options.numLeapfrogSteps,
      logProbGrad,
    );
    state = stepResult.state;
    acceptSum += stepResult.info.acceptProb;
    samples.push(state.position);
  }

  return {
    draws: stackTrees(samples),
    acceptRate: acceptSum / options.numSamples,
    stepSize,
    inverseMassMatrix: invMass,
  };
}

export async function hmc<T extends PyTree>(
  logProbFn: LogProbFn<T>,
  options: HMCOptions<T>,
): Promise<HMCResult<T>> {
  const numWarmup = options.numWarmup ?? 1000;
  const numLeapfrogSteps = options.numLeapfrogSteps ?? 25;
  const numChains = options.numChains ?? 1;
  const initialStepSize = options.initialStepSize ?? 0.1;
  const targetAcceptRate = options.targetAcceptRate ?? 0.8;
  const adaptMassMatrix = options.adaptMassMatrix ?? true;

  const logProbGrad = valueAndGrad(logProbFn);

  const baseInvMass =
    options.inverseMassMatrix ?? (treeOnesLike(options.initialParams) as T);

  const chainKeys = splitKey(options.key, numChains);
  const chainResults = chainKeys.map((chainKey) => {
    const initPosition = tree.ref(options.initialParams) as T;
    const invMass = tree.ref(baseInvMass) as T;
    return runChain(chainKey, initPosition, logProbGrad, {
      numSamples: options.numSamples,
      numWarmup,
      numLeapfrogSteps,
      targetAcceptRate,
      initialStepSize,
      inverseMassMatrix: invMass,
      adaptMassMatrix,
    });
  });

  const chainDraws = chainResults.map((result) => result.draws);
  const draws = stackTrees(chainDraws, 0) as T;

  const acceptRate =
    chainResults.reduce((sum, result) => sum + result.acceptRate, 0) /
    chainResults.length;

  const stepSize =
    chainResults.reduce((sum, result) => sum + result.stepSize, 0) /
    chainResults.length;

  const massMatrix = treeReciprocal(chainResults[0].inverseMassMatrix) as T;

  return {
    draws,
    stats: {
      acceptRate,
      stepSize,
      massMatrix,
    },
  };
}
