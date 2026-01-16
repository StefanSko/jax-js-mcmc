import { numpy as np, tree } from "@jax-js/jax";
import type { JsTree, GradLogProbFn } from "./types";

/**
 * Update position: q = q + eps * M^{-1} * p
 */
function updatePosition<T extends JsTree<np.Array>>(
  q: T,
  p: T,
  stepSize: number,
  massMatrix?: T,
): T {
  if (massMatrix === undefined) {
    return tree.map(
      (qLeaf: np.Array, pLeaf: np.Array) => qLeaf.add(pLeaf.mul(stepSize)),
      tree.ref(q) as JsTree<np.Array>,
      tree.ref(p) as JsTree<np.Array>,
    ) as T;
  }

  return tree.map(
    (qLeaf: np.Array, pLeaf: np.Array, mLeaf: np.Array) =>
      qLeaf.add(pLeaf.div(mLeaf).mul(stepSize)),
    tree.ref(q) as JsTree<np.Array>,
    tree.ref(p) as JsTree<np.Array>,
    tree.ref(massMatrix) as JsTree<np.Array>,
  ) as T;
}

/**
 * Update momentum: p = p + eps * grad_logProb(q)
 */
function updateMomentum<T extends JsTree<np.Array>>(
  p: T,
  grad: T,
  stepSize: number,
): T {
  return tree.map(
    (pLeaf: np.Array, gradLeaf: np.Array) =>
      pLeaf.add(gradLeaf.mul(stepSize)),
    tree.ref(p) as JsTree<np.Array>,
    tree.ref(grad) as JsTree<np.Array>,
  ) as T;
}

/**
 * Leapfrog integrator for Hamiltonian dynamics.
 *
 * Implements the symplectic integrator:
 *   p = p + (eps/2) * grad_logProb(q)     [half step]
 *   for i in 1..numSteps:
 *     q = q + eps * M^{-1} * p            [full step]
 *     p = p + eps * grad_logProb(q)       [full step, half on last]
 *
 * Where grad_logProb is the gradient of the log probability (negative potential energy gradient).
 * Note: grad_logProb = -grad(U), so we ADD the gradient to momentum (since F = -grad(U) = grad(logProb))
 *
 * @param position Initial position (parameter tree)
 * @param momentum Initial momentum (same structure as position)
 * @param gradLogProb Function that computes gradient of log probability
 * @param stepSize Integration step size epsilon
 * @param numSteps Number of leapfrog steps
 * @param massMatrix Optional diagonal mass matrix (same structure as position). If undefined, uses identity.
 * @returns [finalPosition, finalMomentum] after integration
 */
export function leapfrog<T extends JsTree<np.Array>>(
  position: T,
  momentum: T,
  gradLogProb: GradLogProbFn<T>,
  stepSize: number,
  numSteps: number,
  massMatrix?: T,
): [T, T] {
  let q = position;
  let p = momentum;

  // Half step momentum
  const gradInitial = gradLogProb(tree.ref(q) as T);
  p = updateMomentum(p, gradInitial, stepSize * 0.5);

  // Main leapfrog loop
  for (let i = 0; i < numSteps; i++) {
    // Full step position
    q = updatePosition(q, p, stepSize, massMatrix);

    // Momentum step: full step for all but last iteration, half step for last
    const gradCurrent = gradLogProb(tree.ref(q) as T);
    const momentumStepSize = i === numSteps - 1 ? stepSize * 0.5 : stepSize;
    p = updateMomentum(p, gradCurrent, momentumStepSize);
  }

  return [q, p];
}
