export type Vector = number[];

export function add(a: Vector, b: Vector): Vector {
  if (a.length !== b.length) {
    throw new Error("Vector length mismatch");
  }
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    out[i] = a[i] + b[i];
  }
  return out;
}

export function scale(a: Vector, s: number): Vector {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    out[i] = a[i] * s;
  }
  return out;
}

export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) {
    throw new Error("Vector length mismatch");
  }
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function maxAbsDiff(a: Vector, b: Vector): number {
  if (a.length !== b.length) {
    throw new Error("Vector length mismatch");
  }
  let max = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) {
      max = diff;
    }
  }
  return max;
}

export function negate(a: Vector): Vector {
  return scale(a, -1);
}
