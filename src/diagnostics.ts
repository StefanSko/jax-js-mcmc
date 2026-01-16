import { numpy as np, tree, type Array as JaxArray, type JsTree } from "@jax-js/jax";

import { type PyTree } from "./tree";

export function rhat(draws: JaxArray): JaxArray {
  const shape = np.shape(draws) as number[];
  if (shape.length < 2) {
    throw new Error("rhat expects draws shaped [chains, samples, ...]");
  }
  const numChains = shape[0];
  const numSamples = shape[1];
  const correction = numChains > 1 ? 1 : 0;
  const withinCorrection = numSamples > 1 ? 1 : 0;

  const chainMeans = np.mean(draws, 1);
  const overallMean = np.mean(chainMeans, 0);

  const between = np.multiply(
    numSamples,
    np.var_(chainMeans, 0, { mean: overallMean, correction }),
  );
  const within = np.mean(
    np.var_(draws, 1, { correction: withinCorrection }),
    0,
  );

  const varHat = np.add(
    np.multiply((numSamples - 1) / numSamples, within),
    np.multiply(1 / numSamples, between),
  );

  return np.sqrt(np.divide(varHat, np.add(within, 1e-12)));
}

function isArrayLike(value: unknown): value is ArrayLike<number> {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}

function inferShape(value: unknown): number[] {
  if (!isArrayLike(value)) {
    return [];
  }
  const arr = Array.isArray(value) ? value : Array.from(value);
  if (arr.length === 0) {
    return [0];
  }
  return [arr.length, ...inferShape(arr[0])];
}

function flattenValues(value: unknown): number[] {
  if (!isArrayLike(value)) {
    return [Number(value)];
  }
  const arr = Array.isArray(value) ? value : Array.from(value);
  const out: number[] = [];
  for (const item of arr) {
    out.push(...flattenValues(item));
  }
  return out;
}

function unflatten(values: number[], shape: number[]): unknown {
  if (shape.length === 0) {
    return values[0];
  }
  const [dim, ...rest] = shape;
  const stride = rest.reduce((acc, val) => acc * val, 1);
  const out = new Array(dim);
  for (let i = 0; i < dim; i += 1) {
    const start = i * stride;
    const chunk = values.slice(start, start + stride);
    out[i] = unflatten(chunk, rest);
  }
  return out;
}

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

function variance(values: number[], meanValue: number): number {
  if (values.length < 2) {
    return 0;
  }
  let sum = 0;
  for (const value of values) {
    const diff = value - meanValue;
    sum += diff * diff;
  }
  return sum / (values.length - 1);
}

function computeEss(seriesByChain: number[][]): number {
  const numChains = seriesByChain.length;
  const numSamples = seriesByChain[0]?.length ?? 0;
  if (numSamples < 2 || numChains === 0) {
    return numChains * numSamples;
  }

  const means = seriesByChain.map((series) => mean(series));
  const variances = seriesByChain.map((series, index) => variance(series, means[index]));

  const maxLag = Math.min(1000, numSamples - 1);
  const rhos: number[] = [];

  for (let lag = 1; lag <= maxLag; lag += 1) {
    let rhoSum = 0;
    let count = 0;
    for (let chain = 0; chain < numChains; chain += 1) {
      const series = seriesByChain[chain];
      const varValue = variances[chain];
      if (varValue === 0) {
        continue;
      }
      let cov = 0;
      for (let t = 0; t < numSamples - lag; t += 1) {
        cov += (series[t] - means[chain]) * (series[t + lag] - means[chain]);
      }
      cov /= numSamples - 1;
      rhoSum += cov / varValue;
      count += 1;
    }
    if (count === 0) {
      rhos.push(0);
    } else {
      rhos.push(rhoSum / count);
    }
  }

  let rhoSum = 0;
  for (let i = 0; i < rhos.length; i += 2) {
    const pair = rhos[i] + (rhos[i + 1] ?? 0);
    if (pair < 0) {
      break;
    }
    rhoSum += pair;
  }

  return (numChains * numSamples) / (1 + 2 * rhoSum);
}

export function ess(draws: JaxArray): JaxArray {
  const js = draws.js();
  if (!Array.isArray(js)) {
    return np.array(1);
  }
  const numChains = js.length;
  const numSamples = Array.isArray(js[0]) ? js[0].length : 0;
  const sampleShape = inferShape(js[0]?.[0]);
  const flat = js.map((chain) =>
    (chain as unknown[]).map((sample) => flattenValues(sample)),
  );
  const dim = flat[0]?.[0]?.length ?? 0;

  const essValues = new Array(dim).fill(0).map((_, dimIndex) => {
    const seriesByChain = new Array(numChains).fill(0).map((__, chainIndex) => {
      const chain = flat[chainIndex] ?? [];
      return chain.map((sample) => sample[dimIndex]);
    });
    return computeEss(seriesByChain);
  });

  const nested = unflatten(essValues, sampleShape);
  return np.array(nested);
}

export type SummaryStats = {
  mean: unknown;
  sd: unknown;
  q5: unknown;
  q25: unknown;
  q50: unknown;
  q75: unknown;
  q95: unknown;
  rhat: unknown;
  ess: unknown;
};

function quantile(values: number[], q: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (sorted.length - 1) * q;
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) {
    return sorted[lower];
  }
  const weight = idx - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function summarizeLeaf(leaf: JaxArray): SummaryStats {
  const js = leaf.js();
  if (!Array.isArray(js)) {
    return {
      mean: js,
      sd: 0,
      q5: js,
      q25: js,
      q50: js,
      q75: js,
      q95: js,
      rhat: 1,
      ess: 1,
    };
  }

  const numChains = js.length;
  const numSamples = Array.isArray(js[0]) ? js[0].length : 0;
  const sampleShape = inferShape(js[0]?.[0]);
  const flat = js.map((chain) =>
    (chain as unknown[]).map((sample) => flattenValues(sample)),
  );
  const dim = flat[0]?.[0]?.length ?? 0;

  const stats = {
    mean: new Array(dim).fill(0),
    sd: new Array(dim).fill(0),
    q5: new Array(dim).fill(0),
    q25: new Array(dim).fill(0),
    q50: new Array(dim).fill(0),
    q75: new Array(dim).fill(0),
    q95: new Array(dim).fill(0),
  };

  for (let dimIndex = 0; dimIndex < dim; dimIndex += 1) {
    const values: number[] = [];
    for (let chain = 0; chain < numChains; chain += 1) {
      for (let sample = 0; sample < numSamples; sample += 1) {
        values.push(flat[chain][sample][dimIndex]);
      }
    }

    const meanValue = mean(values);
    stats.mean[dimIndex] = meanValue;
    stats.sd[dimIndex] = Math.sqrt(variance(values, meanValue));
    stats.q5[dimIndex] = quantile(values, 0.05);
    stats.q25[dimIndex] = quantile(values, 0.25);
    stats.q50[dimIndex] = quantile(values, 0.5);
    stats.q75[dimIndex] = quantile(values, 0.75);
    stats.q95[dimIndex] = quantile(values, 0.95);
  }

  const rhatValues = rhat(leaf).js();
  const essValues = ess(leaf).js();

  return {
    mean: unflatten(stats.mean, sampleShape),
    sd: unflatten(stats.sd, sampleShape),
    q5: unflatten(stats.q5, sampleShape),
    q25: unflatten(stats.q25, sampleShape),
    q50: unflatten(stats.q50, sampleShape),
    q75: unflatten(stats.q75, sampleShape),
    q95: unflatten(stats.q95, sampleShape),
    rhat: rhatValues,
    ess: essValues,
  };
}

export function summary(draws: PyTree): JsTree<SummaryStats> {
  return tree.map((leaf: JaxArray) => summarizeLeaf(leaf), draws) as JsTree<SummaryStats>;
}
