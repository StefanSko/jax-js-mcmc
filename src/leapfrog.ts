import { numpy as np, tree } from "@jax-js/jax";
import type { JsTree, GradLogProbFn } from "./types";

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
  // CRITICAL: Move semantics
  // - position and momentum will be reassigned in loop
  // - Use tree.ref() when values need to survive operations
  // - gradLogProb returns arrays that may share underlying storage across calls
  //   MUST use tree.ref() on gradient result before passing to tree.map

  let q = position;
  let p = momentum;

  // Half step momentum: p = p + (eps/2) * grad_logProb(q)
  const gradInitial = gradLogProb(tree.ref(q) as T);

  p = tree.map(
    (pLeaf: np.Array, gradLeaf: np.Array) =>
      pLeaf.add(gradLeaf.mul(stepSize * 0.5)),
    tree.ref(p) as JsTree<np.Array>,
    tree.ref(gradInitial) as JsTree<np.Array>, // CRITICAL: use tree.ref on gradient!
  ) as T;

  // Main leapfrog loop
  for (let i = 0; i < numSteps; i++) {
    // Full step position: q = q + eps * M^{-1} * p
    if (massMatrix === undefined) {
      // Identity mass matrix: M^{-1} = I, so just q = q + eps * p
      q = tree.map(
        (qLeaf: np.Array, pLeaf: np.Array) => qLeaf.add(pLeaf.mul(stepSize)),
        tree.ref(q) as JsTree<np.Array>,
        tree.ref(p) as JsTree<np.Array>,
      ) as T;
    } else {
      // Diagonal mass matrix: M^{-1} * p = p / M (elementwise)
      q = tree.map(
        (qLeaf: np.Array, pLeaf: np.Array, mLeaf: np.Array) =>
          qLeaf.add(pLeaf.div(mLeaf).mul(stepSize)),
        tree.ref(q) as JsTree<np.Array>,
        tree.ref(p) as JsTree<np.Array>,
        tree.ref(massMatrix) as JsTree<np.Array>,
      ) as T;
    }

    // Momentum step: full step for all but last iteration, half step for last
    const gradCurrent = gradLogProb(tree.ref(q) as T);
    const momentumStepSize = i === numSteps - 1 ? stepSize * 0.5 : stepSize;

    p = tree.map(
      (pLeaf: np.Array, gradLeaf: np.Array) =>
        pLeaf.add(gradLeaf.mul(momentumStepSize)),
      tree.ref(p) as JsTree<np.Array>,
      tree.ref(gradCurrent) as JsTree<np.Array>, // CRITICAL: use tree.ref on gradient!
    ) as T;
  }

  return [q, p];
}
