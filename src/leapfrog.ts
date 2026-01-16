export type Vector = number[];

export function leapfrog(
  position: Vector,
  momentum: Vector,
  gradPotential: (q: Vector) => Vector,
  stepSize: number,
  numSteps: number,
): [Vector, Vector] {
  const q = position.slice();
  const p = momentum.slice();

  if (q.length !== p.length) {
    throw new Error("position and momentum must have the same length");
  }
  if (numSteps <= 0 || stepSize === 0) {
    return [q, p];
  }

  let grad = gradPotential(q);
  if (grad.length !== q.length) {
    throw new Error("gradPotential must return a vector of the same length");
  }

  for (let i = 0; i < p.length; i += 1) {
    p[i] -= 0.5 * stepSize * grad[i];
  }

  for (let step = 0; step < numSteps; step += 1) {
    for (let i = 0; i < q.length; i += 1) {
      q[i] += stepSize * p[i];
    }

    grad = gradPotential(q);
    if (grad.length !== q.length) {
      throw new Error("gradPotential must return a vector of the same length");
    }

    const scale = step === numSteps - 1 ? 0.5 : 1.0;
    for (let i = 0; i < p.length; i += 1) {
      p[i] -= scale * stepSize * grad[i];
    }
  }

  return [q, p];
}
