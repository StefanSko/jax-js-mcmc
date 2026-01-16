/**
 * MCMC Diagnostics
 *
 * Provides R-hat (Gelman-Rubin) and ESS (effective sample size) diagnostics.
 */

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

  // Compute chain means
  const chainMeans = splitChains.map(
    (chain) => chain.reduce((a, b) => a + b, 0) / chain.length,
  );

  // Compute overall mean
  const overallMean = chainMeans.reduce((a, b) => a + b, 0) / numChains;

  // Compute between-chain variance B
  let B = 0;
  for (const mean of chainMeans) {
    B += (mean - overallMean) ** 2;
  }
  B = (B * n) / (numChains - 1);

  // Compute within-chain variance W
  let W = 0;
  for (let i = 0; i < numChains; i++) {
    const chain = splitChains[i];
    const mean = chainMeans[i];
    let s2 = 0;
    for (const x of chain) {
      s2 += (x - mean) ** 2;
    }
    W += s2 / (n - 1);
  }
  W /= numChains;

  // Compute pooled variance estimate
  const varPlus = ((n - 1) / n) * W + B / n;

  // R-hat is sqrt(varPlus / W)
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
  // Flatten all chains
  const allSamples: number[] = [];
  for (const chain of draws) {
    allSamples.push(...chain);
  }

  const n = allSamples.length;
  const mean = allSamples.reduce((a, b) => a + b, 0) / n;

  // Compute variance
  let variance = 0;
  for (const x of allSamples) {
    variance += (x - mean) ** 2;
  }
  variance /= n - 1;

  if (variance < 1e-10) {
    return n; // No variance, all samples are the same
  }

  // Compute autocorrelation using FFT-like approach
  // For simplicity, use direct computation up to maxLag
  const maxLag = Math.min(n - 1, Math.floor(n / 2));
  const rho: number[] = [];

  for (let lag = 0; lag <= maxLag; lag++) {
    let autocorr = 0;
    for (let i = 0; i < n - lag; i++) {
      autocorr += (allSamples[i] - mean) * (allSamples[i + lag] - mean);
    }
    rho.push(autocorr / (n * variance));
  }

  // Geyer's initial monotone sequence estimator
  // Sum pairs of consecutive autocorrelations until they become negative
  let essSum = rho[0]; // Start with rho(0) = 1

  for (let t = 1; t < maxLag - 1; t += 2) {
    const pairSum = rho[t] + rho[t + 1];
    if (pairSum < 0) {
      break;
    }
    essSum += 2 * pairSum;
  }

  // ESS = n / (1 + 2 * sum of autocorrelations)
  const effectiveSampleSize = n / essSum;

  return Math.max(1, effectiveSampleSize);
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
): {
  params: Array<{
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
  }>;
} {
  const numChains = draws.length;
  const numSamples = draws[0].length;
  const numParams = draws[0][0].length;

  const params: Array<{
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
  }> = [];

  for (let p = 0; p < numParams; p++) {
    // Extract draws for this parameter across all chains
    const paramDraws: number[][] = [];
    for (let c = 0; c < numChains; c++) {
      const chainDraws: number[] = [];
      for (let s = 0; s < numSamples; s++) {
        chainDraws.push(draws[c][s][p]);
      }
      paramDraws.push(chainDraws);
    }

    // Flatten for summary statistics
    const allDraws: number[] = [];
    for (const chain of paramDraws) {
      allDraws.push(...chain);
    }

    // Compute statistics
    const n = allDraws.length;
    const mean = allDraws.reduce((a, b) => a + b, 0) / n;
    const std = Math.sqrt(
      allDraws.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1),
    );

    // Quantiles
    const sorted = [...allDraws].sort((a, b) => a - b);
    const q5 = sorted[Math.floor(n * 0.05)];
    const q25 = sorted[Math.floor(n * 0.25)];
    const q50 = sorted[Math.floor(n * 0.5)];
    const q75 = sorted[Math.floor(n * 0.75)];
    const q95 = sorted[Math.floor(n * 0.95)];

    // Diagnostics
    const paramRhat = rhat(paramDraws);
    const paramEss = ess(paramDraws);

    params.push({
      name: paramNames?.[p] ?? `param_${p}`,
      mean,
      std,
      q5,
      q25,
      q50,
      q75,
      q95,
      rhat: paramRhat,
      ess: paramEss,
    });
  }

  return { params };
}
