import type { HMCOptions, HMCResult, LogProbFn } from "./types.js";

/**
 * Hamiltonian Monte Carlo sampler.
 *
 * @param logProb - Log probability function (returns scalar Array)
 * @param options - HMC configuration options
 * @returns HMC result with draws and statistics
 */
export async function hmc(
  _logProb: LogProbFn,
  _options: HMCOptions,
): Promise<HMCResult> {
  // TODO: Implement after leapfrog and posterior tests pass
  throw new Error("hmc not implemented");
}
