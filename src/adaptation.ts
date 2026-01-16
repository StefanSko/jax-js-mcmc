import { type Array as JaxArray } from "@jax-js/jax";

import {
  treeAdd,
  treeAddScalar,
  treeDivScalar,
  treeMul,
  treeOnesLike,
  treeReciprocal,
  treeScale,
  treeSub,
  treeZerosLike,
  type PyTree,
} from "./tree";

export type DualAveragingState = {
  logStep: number;
  logStepAvg: number;
  hBar: number;
  t: number;
  mu: number;
};

export function initDualAveraging(initialStepSize: number): DualAveragingState {
  const logStep = Math.log(initialStepSize);
  return {
    logStep,
    logStepAvg: logStep,
    hBar: 0,
    t: 0,
    mu: Math.log(10 * initialStepSize),
  };
}

export function updateDualAveraging(
  state: DualAveragingState,
  acceptProb: number,
  targetAcceptRate: number,
  gamma = 0.05,
  t0 = 10,
  kappa = 0.75,
): DualAveragingState {
  const t = state.t + 1;
  const eta = 1 / (t + t0);
  const hBar = (1 - eta) * state.hBar + eta * (targetAcceptRate - acceptProb);
  const logStep = state.mu - (Math.sqrt(t) / gamma) * hBar;
  const w = Math.pow(t, -kappa);
  const logStepAvg = w * logStep + (1 - w) * state.logStepAvg;

  return {
    logStep,
    logStepAvg,
    hBar,
    t,
    mu: state.mu,
  };
}

export type MassMatrixState<T extends PyTree> = {
  mean: T;
  m2: T;
  count: number;
};

export function initMassMatrix<T extends PyTree>(reference: T): MassMatrixState<T> {
  return {
    mean: treeZerosLike(reference) as T,
    m2: treeZerosLike(reference) as T,
    count: 0,
  };
}

export function updateMassMatrix<T extends PyTree>(
  state: MassMatrixState<T>,
  sample: T,
): MassMatrixState<T> {
  const count = state.count + 1;
  const delta = treeSub(sample, state.mean);
  const mean = treeAdd(state.mean, treeDivScalar(delta, count));
  const delta2 = treeSub(sample, mean);
  const m2 = treeAdd(state.m2, treeMul(delta, delta2));

  return {
    mean,
    m2,
    count,
  };
}

export function finalizeMassMatrix<T extends PyTree>(
  state: MassMatrixState<T>,
  jitter = 1e-3,
): T {
  if (state.count < 2) {
    return treeOnesLike(state.mean) as T;
  }
  const variance = treeDivScalar(state.m2, state.count - 1);
  const stabilized = treeAddScalar(variance, jitter);
  return treeReciprocal(stabilized) as T;
}

export type WarmupWindow = {
  size: number;
  adaptMassMatrix: boolean;
};

export function buildWarmupSchedule(numWarmup: number): WarmupWindow[] {
  if (numWarmup <= 0) {
    return [];
  }
  const initialBuffer = Math.max(1, Math.floor(numWarmup * 0.15));
  const finalBuffer = Math.max(1, Math.floor(numWarmup * 0.1));
  let remaining = Math.max(0, numWarmup - initialBuffer - finalBuffer);

  const windows: WarmupWindow[] = [];
  if (initialBuffer > 0) {
    windows.push({ size: initialBuffer, adaptMassMatrix: false });
  }

  let windowSize = 25;
  while (remaining > 0) {
    const size = Math.min(remaining, windowSize);
    windows.push({ size, adaptMassMatrix: true });
    remaining -= size;
    windowSize *= 2;
  }

  if (finalBuffer > 0) {
    windows.push({ size: finalBuffer, adaptMassMatrix: false });
  }

  return windows;
}

export function clampStepSize(stepSize: number, min = 1e-4, max = 1e2): number {
  if (!Number.isFinite(stepSize)) {
    return min;
  }
  return Math.min(Math.max(stepSize, min), max);
}

export type AcceptProbEvaluator = (key: JaxArray, stepSize: number) => {
  acceptProb: number;
  nextKey: JaxArray;
};

export function findReasonableStepSize(
  key: JaxArray,
  evaluate: AcceptProbEvaluator,
  targetAcceptRate: number,
  initialStepSize = 1.0,
  maxIterations = 20,
): { stepSize: number; key: JaxArray } {
  let stepSize = initialStepSize;
  let currentKey = key;

  const first = evaluate(currentKey, stepSize);
  currentKey = first.nextKey;
  let direction = first.acceptProb > targetAcceptRate ? 1 : -1;

  for (let i = 0; i < maxIterations; i += 1) {
    const scale = direction > 0 ? 2.0 : 0.5;
    stepSize = clampStepSize(stepSize * scale);
    const result = evaluate(currentKey, stepSize);
    currentKey = result.nextKey;

    if (direction > 0 && result.acceptProb < targetAcceptRate) {
      break;
    }
    if (direction < 0 && result.acceptProb > targetAcceptRate) {
      break;
    }
  }

  return { stepSize, key: currentKey };
}
