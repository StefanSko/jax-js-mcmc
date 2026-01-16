import type { Array, JsTree } from "@jax-js/jax";
import { tree } from "@jax-js/jax";
import { prod } from "./tree-utils";

function reshape1DToNested(data: number[], shape: number[]): any {
  if (shape.length === 0) return data[0];
  const [dim, ...rest] = shape;
  const restSize = rest.reduce((a, b) => a * b, 1);
  const out = [] as any[];
  for (let i = 0; i < dim; i++) {
    const start = i * restSize;
    const end = start + restSize;
    out.push(reshape1DToNested(data.slice(start, end), rest));
  }
  return out;
}

function getIndex(
  chain: number,
  sample: number,
  param: number,
  numSamples: number,
  paramSize: number,
): number {
  return (chain * numSamples + sample) * paramSize + param;
}

function computeRhatFlat(draws: Array): number[] {
  const shape = draws.shape;
  if (shape.length < 2) {
    throw new Error("rhat expects shape [chains, samples, ...]");
  }
  const m = shape[0];
  const n = shape[1];
  const nEven = n - (n % 2);
  const half = nEven / 2;
  const paramShape = shape.slice(2);
  const paramSize = prod(paramShape.length ? paramShape : [1]);
  const data = draws.dataSync();

  if (nEven < 2 || m < 1) {
    return new Array(paramSize).fill(NaN);
  }

  const rhat = new Array(paramSize).fill(0);
  const mSplit = m * 2;

  for (let p = 0; p < paramSize; p++) {
    const means = new Array(mSplit).fill(0);
    const vars = new Array(mSplit).fill(0);

    for (let chain = 0; chain < m; chain++) {
      for (let halfIdx = 0; halfIdx < 2; halfIdx++) {
        const start = halfIdx * half;
        let sum = 0;
        for (let i = 0; i < half; i++) {
          const idx = getIndex(chain, start + i, p, n, paramSize);
          sum += data[idx];
        }
        const mean = sum / half;
        means[chain * 2 + halfIdx] = mean;

        let varSum = 0;
        for (let i = 0; i < half; i++) {
          const idx = getIndex(chain, start + i, p, n, paramSize);
          const diff = data[idx] - mean;
          varSum += diff * diff;
        }
        vars[chain * 2 + halfIdx] = varSum / (half - 1);
      }
    }

    const meanOfMeans = means.reduce((a, b) => a + b, 0) / mSplit;
    const varMeans =
      means.reduce((acc, v) => acc + (v - meanOfMeans) ** 2, 0) /
      (mSplit - 1);
    const W = vars.reduce((a, b) => a + b, 0) / mSplit;
    const B = half * varMeans;
    const varHat = ((half - 1) / half) * W + B / half;
    rhat[p] = Math.sqrt(varHat / W);
  }

  return rhat;
}

function computeEssFlat(draws: Array): number[] {
  const shape = draws.shape;
  if (shape.length < 2) {
    throw new Error("ess expects shape [chains, samples, ...]");
  }
  const m = shape[0];
  const n = shape[1];
  const paramShape = shape.slice(2);
  const paramSize = prod(paramShape.length ? paramShape : [1]);
  const data = draws.dataSync();

  const ess = new Array(paramSize).fill(0);

  for (let p = 0; p < paramSize; p++) {
    const chainMeans = new Array(m).fill(0);
    const chainVars = new Array(m).fill(0);

    for (let chain = 0; chain < m; chain++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += data[getIndex(chain, i, p, n, paramSize)];
      }
      const mean = sum / n;
      chainMeans[chain] = mean;

      let varSum = 0;
      for (let i = 0; i < n; i++) {
        const diff = data[getIndex(chain, i, p, n, paramSize)] - mean;
        varSum += diff * diff;
      }
      chainVars[chain] = varSum / (n - 1);
    }

    const W = chainVars.reduce((a, b) => a + b, 0) / m;
    if (!Number.isFinite(W) || W <= 0) {
      ess[p] = NaN;
      continue;
    }

    const rho: number[] = [];
    for (let t = 1; t < n; t++) {
      let autocov = 0;
      for (let chain = 0; chain < m; chain++) {
        const mean = chainMeans[chain];
        let chainSum = 0;
        for (let i = 0; i < n - t; i++) {
          const x0 = data[getIndex(chain, i, p, n, paramSize)] - mean;
          const x1 = data[getIndex(chain, i + t, p, n, paramSize)] - mean;
          chainSum += x0 * x1;
        }
        autocov += chainSum / (n - t);
      }
      autocov /= m;
      rho.push(autocov / W);
    }

    let rhoSum = 0;
    for (let t = 0; t < rho.length; t += 2) {
      const pair = rho[t] + (rho[t + 1] ?? 0);
      if (pair < 0) break;
      rhoSum += pair;
    }

    const eff = (m * n) / (1 + 2 * rhoSum);
    ess[p] = Math.min(eff, m * n);
  }

  return ess;
}

export function rhat(draws: Array): any {
  const paramShape = draws.shape.slice(2);
  const flat = computeRhatFlat(draws);
  return reshape1DToNested(flat, paramShape);
}

export function ess(draws: Array): any {
  const paramShape = draws.shape.slice(2);
  const flat = computeEssFlat(draws);
  return reshape1DToNested(flat, paramShape);
}

export function summary(drawsTree: JsTree<Array>): any {
  const [leaves, treedef] = tree.flatten(drawsTree);
  const summaries = leaves.map((draws) => summaryArray(draws));
  return tree.unflatten(treedef, summaries);
}

function summaryArray(draws: Array) {
  const shape = draws.shape;
  if (shape.length < 2) {
    throw new Error("summary expects shape [chains, samples, ...]");
  }
  const m = shape[0];
  const n = shape[1];
  const paramShape = shape.slice(2);
  const paramSize = prod(paramShape.length ? paramShape : [1]);
  const data = draws.dataSync();

  const total = m * n;
  const means = new Array(paramSize).fill(0);
  const sds = new Array(paramSize).fill(0);
  const q5 = new Array(paramSize).fill(0);
  const q25 = new Array(paramSize).fill(0);
  const q50 = new Array(paramSize).fill(0);
  const q75 = new Array(paramSize).fill(0);
  const q95 = new Array(paramSize).fill(0);

  for (let p = 0; p < paramSize; p++) {
    let sum = 0;
    for (let chain = 0; chain < m; chain++) {
      for (let i = 0; i < n; i++) {
        sum += data[getIndex(chain, i, p, n, paramSize)];
      }
    }
    const mean = sum / total;
    means[p] = mean;

    let varSum = 0;
    const samples = new Array(total);
    let idx = 0;
    for (let chain = 0; chain < m; chain++) {
      for (let i = 0; i < n; i++) {
        const v = data[getIndex(chain, i, p, n, paramSize)];
        const diff = v - mean;
        varSum += diff * diff;
        samples[idx++] = v;
      }
    }
    sds[p] = Math.sqrt(varSum / (total - 1));

    samples.sort((a, b) => a - b);
    const pick = (q: number) => samples[Math.floor(q * (total - 1))];
    q5[p] = pick(0.05);
    q25[p] = pick(0.25);
    q50[p] = pick(0.5);
    q75[p] = pick(0.75);
    q95[p] = pick(0.95);
  }

  const rhatVals = computeRhatFlat(draws);
  const essVals = computeEssFlat(draws);

  return {
    mean: reshape1DToNested(means, paramShape),
    sd: reshape1DToNested(sds, paramShape),
    q5: reshape1DToNested(q5, paramShape),
    q25: reshape1DToNested(q25, paramShape),
    q50: reshape1DToNested(q50, paramShape),
    q75: reshape1DToNested(q75, paramShape),
    q95: reshape1DToNested(q95, paramShape),
    rhat: reshape1DToNested(rhatVals, paramShape),
    ess: reshape1DToNested(essVals, paramShape),
  };
}
