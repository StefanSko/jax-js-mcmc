import type { Array, JsTree } from "@jax-js/jax";

// Re-export JsTree for convenience
export type { JsTree };

/**
 * PRNG key type - an Array with shape [..., 2] containing uint32 values.
 */
export type PRNGKey = Array;

/**
 * Log probability function.
 * Takes parameters as a JsTree<Array> and returns a scalar Array (0-dim float32).
 */
export type LogProbFn = (params: JsTree<Array>) => Array;

/**
 * Gradient of log probability function.
 * Returns gradients with same structure as input params.
 */
export type GradLogProbFn = (params: JsTree<Array>) => JsTree<Array>;

/**
 * HMC sampler options.
 */
export interface HMCOptions {
  /** Initial parameter values (required) */
  initialParams: JsTree<Array>;
  /** PRNG key for randomness (required) */
  key: Array;
  /** Number of samples to draw after warmup (required) */
  numSamples: number;

  /** Number of warmup iterations for adaptation. Default: 1000 */
  numWarmup?: number;
  /** Number of leapfrog steps per HMC iteration. Default: 25 */
  numLeapfrogSteps?: number;
  /** Number of independent chains. Default: 1 */
  numChains?: number;
  /** Initial step size (may be adjusted by init heuristic). Default: 0.1 */
  initialStepSize?: number;
  /** Target acceptance rate for step size adaptation. Default: 0.8 */
  targetAcceptRate?: number;
  /** Whether to adapt the diagonal mass matrix. Default: true */
  adaptMassMatrix?: boolean;
}

/**
 * HMC sampler result.
 */
export interface HMCResult {
  /** Posterior samples. Shape: [numChains, numSamples, ...paramShape] */
  draws: JsTree<Array>;
  /** Sampler statistics */
  stats: HMCStats;
}

/**
 * HMC sampler statistics.
 */
export interface HMCStats {
  /** Acceptance rate per chain. Shape: [numChains] */
  acceptRate: Array;
  /** Final step size per chain. Shape: [numChains] */
  stepSize: Array;
  /** Final diagonal mass matrix per chain (if adapted) */
  massMatrix?: JsTree<Array>;
}

/**
 * Summary statistics for a single parameter.
 */
export interface ParamSummary {
  mean: number;
  sd: number;
  median: number;
  q5: number;
  q25: number;
  q75: number;
  q95: number;
  rhat: number;
  ess: number;
}

// --- Internal types ---

/**
 * Dual averaging state for step size adaptation (Nesterov).
 */
export interface DualAverageState {
  /** Current log step size */
  logStepSize: number;
  /** Averaged log step size (used after warmup) */
  logStepSizeAvg: number;
  /** Cumulative sum of (targetAcceptRate - acceptProb) */
  hSum: number;
  /** Current iteration (1-indexed) */
  iteration: number;
  /** Center for dual averaging: log(10 * initialStepSize) */
  mu: number;
}

/**
 * Dual averaging hyperparameters.
 */
export const DUAL_AVERAGE_PARAMS = {
  gamma: 0.05,
  t0: 10,
  kappa: 0.75,
} as const;

/**
 * Welford online variance state for mass matrix adaptation.
 */
export interface WelfordState {
  /** Sample count */
  count: number;
  /** Running mean */
  mean: JsTree<Array>;
  /** Running M2 (sum of squared deviations) */
  m2: JsTree<Array>;
}

/**
 * Mass matrix jitter for numerical stability.
 */
export const MASS_MATRIX_JITTER = 1e-5;

/**
 * Step size bounds for initialization heuristic.
 */
export const STEP_SIZE_BOUNDS = {
  min: 1e-4,
  max: 1.0,
} as const;
