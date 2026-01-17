import type { Array, JsTree } from "@jax-js/jax";
import { numpy as np, grad } from "@jax-js/jax";
import { leapfrog } from "./leapfrog";
import { hamiltonian } from "./hamiltonian";
import {
  mapTree,
  treeClone,
  treeDispose,
  treeOnesLike,
  treeRef,
  treeZerosLike,
} from "./tree-utils";
import { sampleNormalTree } from "./random-utils";

export interface DualAverageState {
  logStepSize: number;
  logStepSizeAvg: number;
  hSum: number;
  iteration: number;
  mu: number;
  gamma: number;
  t0: number;
  kappa: number;
}

export function initDualAverage(initialStepSize: number): DualAverageState {
  const logStep = Math.log(initialStepSize);
  return {
    logStepSize: logStep,
    logStepSizeAvg: logStep,
    hSum: 0,
    iteration: 0,
    mu: Math.log(10 * initialStepSize),
    gamma: 0.05,
    t0: 10,
    kappa: 0.75,
  };
}

export function updateDualAverage(
  state: DualAverageState,
  acceptProb: number,
  targetAcceptRate: number,
): DualAverageState {
  const iter = state.iteration + 1;
  const eta = 1 / (iter + state.t0);
  const hSum = (1 - eta) * state.hSum + eta * (targetAcceptRate - acceptProb);
  const logStep = state.mu - (Math.sqrt(iter) / state.gamma) * hSum;
  const logStepAvg =
    Math.pow(iter, -state.kappa) * logStep +
    (1 - Math.pow(iter, -state.kappa)) * state.logStepSizeAvg;
  return {
    ...state,
    logStepSize: logStep,
    logStepSizeAvg: logStepAvg,
    hSum,
    iteration: iter,
  };
}

export function clampStepSize(stepSize: number): number {
  return Math.min(1, Math.max(1e-4, stepSize));
}

export interface MassMatrixState<Params extends JsTree<Array>> {
  mean: Params;
  m2: Params;
  count: number;
}

export function initMassMatrixState<Params extends JsTree<Array>>(
  params: Params,
): MassMatrixState<Params> {
  return {
    mean: treeZerosLike(params),
    m2: treeZerosLike(params),
    count: 0,
  };
}

export function updateMassMatrix<Params extends JsTree<Array>>(
  state: MassMatrixState<Params>,
  sample: Params,
): MassMatrixState<Params> {
  const count = state.count + 1;
  const delta = mapTree(
    (x: Array, mean: Array) => x.sub(mean.ref),
    treeRef(sample),
    state.mean,
  ) as Params;
  const mean = mapTree(
    (mean: Array, d: Array) => mean.add(d.ref.div(count)),
    state.mean,
    delta,
  ) as Params;
  const delta2 = mapTree(
    (x: Array, meanNow: Array) => x.sub(meanNow.ref),
    treeRef(sample),
    mean,
  ) as Params;
  const m2 = mapTree(
    (m2: Array, d1: Array, d2: Array) => m2.add(d1.ref.mul(d2.ref)),
    state.m2,
    delta,
    delta2,
  ) as Params;
  return { mean, m2, count };
}

export function finalizeMassMatrix<Params extends JsTree<Array>>(
  state: MassMatrixState<Params>,
  fallback: Params,
): Params {
  if (state.count < 2) {
    return fallback;
  }
  const denom = state.count - 1;
  return mapTree(
    (m2: Array) => m2.div(denom).add(1e-5),
    state.m2,
  ) as Params;
}

function computeAcceptProb<Params extends JsTree<Array>>(
  logProb: (p: Params) => Array,
  gradLogProb: (p: Params) => Params,
  position: Params,
  momentum: Params,
  stepSize: number,
  numSteps: number,
  massMatrix: Params,
): number {
  const h0 = hamiltonian(position, momentum, logProb, massMatrix);
  const [qNew, pNew] = leapfrog(
    treeClone(position),
    momentum,
    gradLogProb,
    stepSize,
    numSteps,
    massMatrix,
  );
  const h1 = hamiltonian(qNew, pNew, logProb, massMatrix);
  const logAccept = h0.sub(h1).item();
  treeDispose(position);
  treeDispose(qNew);
  treeDispose(pNew);
  return Math.min(1, Math.exp(logAccept));
}

export function findReasonableStepSize<Params extends JsTree<Array>>(
  logProb: (p: Params) => Array,
  position: Params,
  massMatrix: Params,
  initialStepSize: number,
  key: Array,
  maxIters: number = 10,
): { stepSize: number; nextKey: Array } {
  const gradLogProb = grad(logProb) as (p: Params) => Params;
  let stepSize = clampStepSize(initialStepSize);
  let currentKey = key;

  for (let i = 0; i < maxIters; i++) {
    const { nextKey, sample } = sampleNormalTree(currentKey, massMatrix);
    currentKey = nextKey;
    const momentum = mapTree(
      (z: Array, m: Array) => z.mul(np.sqrt(m.ref)),
      sample,
      massMatrix,
    ) as Params;
    const acceptProb = computeAcceptProb(
      logProb,
      gradLogProb,
      treeClone(position),
      momentum,
      stepSize,
      1,
      massMatrix,
    );

    if (acceptProb > 0.8 && stepSize < 1) {
      stepSize = clampStepSize(stepSize * 2);
      continue;
    }
    if (acceptProb < 0.2 && stepSize > 1e-4) {
      stepSize = clampStepSize(stepSize * 0.5);
      continue;
    }
    break;
  }
  return { stepSize, nextKey: currentKey };
}

export function initMassMatrix<Params extends JsTree<Array>>(params: Params): Params {
  return treeOnesLike(params);
}
