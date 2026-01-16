/**
 * MCMC Diagnostics
 *
 * Provides R-hat (Gelman-Rubin) and ESS (effective sample size) diagnostics.
 */

/**
 * Compute mean of an array.
 */
function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Compute variance of an array (sample variance with n-1 denominator).
 */
function variance(values: number[], meanValue?: number): number {
  const m = meanValue ?? mean(values);
  return values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1);
}

/**
 * Compute quantile of a sorted array.
 */
function quantile(sorted: number[], q: number): number {
  return sorted[Math.floor(sorted.length * q)];
}

/**
 * Compute split R-hat (Gelman-Rubin statistic) for convergence diagnostics.
 *
 * R-hat compares within-chain and between-chain variance to assess convergence.
 * Values close to 1.0 indicate convergence; values > 1.01 suggest non-convergence.
 *
 * Uses the "split" version that splits each chain in half for more sensitivity.
 *
 * @param draws Array of shape [numChains, numSamples] for a single parameter
 * @returns R-hat statistic
 */
export function rhat(draws: number[][]): number {
  // Split each chain in half
  const splitChains: number[][] = [];
  for (const chain of draws) {
    const mid = Math.floor(chain.length / 2);
    splitChains.push(chain.slice(0, mid));
    splitChains.push(chain.slice(mid));
  }

  const numChains = splitChains.length;
  const n = splitChains[0].length;

  // Compute chain means and overall mean
  const chainMeans = splitChains.map(mean);
  const overallMean = mean(chainMeans);

  // Compute between-chain variance B
  let B = 0;
  for (const m of chainMeans) {
    B += (m - overallMean) ** 2;
  }
  B = (B * n) / (numChains - 1);

  // Compute within-chain variance W
  let W = 0;
  for (let i = 0; i < numChains; i++) {
    W += variance(splitChains[i], chainMeans[i]);
  }
  W /= numChains;

  // Compute pooled variance estimate and R-hat
  const varPlus = ((n - 1) / n) * W + B / n;
  return Math.sqrt(varPlus / W);
}

/**
 * Compute effective sample size (ESS) using Geyer's initial monotone sequence.
 *
 * ESS estimates the number of independent samples, accounting for autocorrelation.
 *
 * @param draws Array of shape [numChains, numSamples] for a single parameter
 * @returns Effective sample size
 */
export function ess(draws: number[][]): number {
  const chainEss = draws.map((chain) => {
    const n = chain.length;
    if (n < 2) {
      return n;
    }

    const sampleMean = mean(chain);
    const sampleVariance = variance(chain, sampleMean);
    if (sampleVariance < 1e-10) {
      return n; // No variance, all samples are the same
    }

    // Convert sample variance (n-1) to population variance (n) for rho[0] = 1.
    const varianceN = sampleVariance * ((n - 1) / n);

    // Compute autocorrelation up to maxLag
    const maxLag = Math.min(n - 1, Math.floor(n / 2));
    const rho: number[] = [];

    for (let lag = 0; lag <= maxLag; lag++) {
      let autocorr = 0;
      for (let i = 0; i < n - lag; i++) {
        autocorr += (chain[i] - sampleMean) * (chain[i + lag] - sampleMean);
      }
      rho.push(autocorr / (n * varianceN));
    }

    // Geyer's initial monotone sequence estimator
    // Sum pairs of consecutive autocorrelations until they become negative
    let essSum = rho[0];
    for (let t = 1; t < maxLag - 1; t += 2) {
      const pairSum = rho[t] + rho[t + 1];
      if (pairSum < 0) {
        break;
      }
      essSum += 2 * pairSum;
    }

    return Math.max(1, n / essSum);
  });

  return chainEss.reduce((a, b) => a + b, 0);
}

/**
 * Summary statistics for a single parameter.
 */
interface ParamSummary {
  name: string;
  mean: number;
  std: number;
  q5: number;
  q25: number;
  q50: number;
  q75: number;
  q95: number;
  rhat: number;
  ess: number;
}

/**
 * Compute summary statistics for MCMC draws.
 *
 * @param draws Array of shape [numChains, numSamples, numParams]
 * @param paramNames Optional parameter names
 * @returns Summary object with mean, std, quantiles, R-hat, and ESS per parameter
 */
export function summary(
  draws: number[][][],
  paramNames?: string[],
): { params: ParamSummary[] } {
  const numChains = draws.length;
  const numParams = draws[0][0].length;

  const params: ParamSummary[] = [];

  for (let p = 0; p < numParams; p++) {
    // Extract draws for this parameter across all chains
    const paramDraws: number[][] = [];
    for (let c = 0; c < numChains; c++) {
      paramDraws.push(draws[c].map((sample) => sample[p]));
    }

    // Flatten and compute statistics
    const allDraws = paramDraws.flat();
    const paramMean = mean(allDraws);
    const std = Math.sqrt(variance(allDraws, paramMean));

    // Quantiles
    const sorted = [...allDraws].sort((a, b) => a - b);

    params.push({
      name: paramNames?.[p] ?? `param_${p}`,
      mean: paramMean,
      std,
      q5: quantile(sorted, 0.05),
      q25: quantile(sorted, 0.25),
      q50: quantile(sorted, 0.5),
      q75: quantile(sorted, 0.75),
      q95: quantile(sorted, 0.95),
      rhat: rhat(paramDraws),
      ess: ess(paramDraws),
    });
  }

  return { params };
}
