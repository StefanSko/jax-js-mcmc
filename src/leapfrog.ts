import { type Array, type JsTree, tree } from "@jax-js/jax";
import type { GradLogProbFn } from "./types.js";

/**
 * Leapfrog integrator for Hamiltonian Monte Carlo.
 *
 * The leapfrog algorithm is a symplectic integrator that preserves:
 * 1. Time-reversibility
 * 2. Phase space volume (Jacobian det = 1)
 * 3. Energy up to O(ε²) per step
 *
 * Algorithm:
 *   p_{1/2} = p_0 + (ε/2) * ∇logProb(q_0)
 *   for i = 1 to L-1:
 *     q_i = q_{i-1} + ε * M^{-1} * p_{i-1/2}
 *     p_{i+1/2} = p_{i-1/2} + ε * ∇logProb(q_i)
 *   q_L = q_{L-1} + ε * M^{-1} * p_{L-1/2}
 *   p_L = p_{L-1/2} + (ε/2) * ∇logProb(q_L)
 *
 * @param position - Current position (parameters) as JsTree<Array>
 * @param momentum - Current momentum as JsTree<Array>
 * @param gradLogProb - Gradient of log probability function
 * @param stepSize - Integration step size (ε)
 * @param numSteps - Number of leapfrog steps (L)
 * @param massMatrix - Optional diagonal mass matrix (defaults to identity)
 * @returns [newPosition, newMomentum]
 */
export function leapfrog(
  position: JsTree<Array>,
  momentum: JsTree<Array>,
  gradLogProb: GradLogProbFn,
  stepSize: number,
  numSteps: number,
  massMatrix?: JsTree<Array>,
): [JsTree<Array>, JsTree<Array>] {
  const halfStep = stepSize / 2;

  // Helper to scale momentum by inverse mass matrix: M^{-1} * p
  // For diagonal mass matrix, this is element-wise division
  function scaleMomentum(p: JsTree<Array>): JsTree<Array> {
    if (!massMatrix) {
      return p; // Identity mass matrix - no scaling needed
    }
    return tree.map(
      (pLeaf: Array, mLeaf: Array) => pLeaf.div(mLeaf) as Array,
      p,
      tree.ref(massMatrix),
    );
  }

  // Helper to add scaled gradient to momentum: p + scale * grad
  function addScaledGrad(
    p: JsTree<Array>,
    grad: JsTree<Array>,
    scale: number,
  ): JsTree<Array> {
    return tree.map(
      (pLeaf: Array, gLeaf: Array) => pLeaf.add(gLeaf.mul(scale)) as Array,
      p,
      grad,
    );
  }

  // Helper to add scaled momentum to position: q + scale * (M^{-1} * p)
  function addScaledMomentum(
    q: JsTree<Array>,
    p: JsTree<Array>,
    scale: number,
  ): JsTree<Array> {
    const scaledP = scaleMomentum(p);
    return tree.map(
      (qLeaf: Array, pLeaf: Array) => qLeaf.add(pLeaf.mul(scale)) as Array,
      q,
      scaledP,
    );
  }

  let q = position;
  let p = momentum;

  // Initial half-step for momentum
  const grad0 = gradLogProb(tree.ref(q));
  p = addScaledGrad(p, grad0, halfStep);

  // Full steps
  for (let i = 0; i < numSteps - 1; i++) {
    // Full step for position
    q = addScaledMomentum(q, tree.ref(p), stepSize);

    // Full step for momentum (except last)
    const grad = gradLogProb(tree.ref(q));
    p = addScaledGrad(p, grad, stepSize);
  }

  // Final full step for position
  q = addScaledMomentum(q, tree.ref(p), stepSize);

  // Final half-step for momentum
  // Use tree.ref(q) since we need to return q
  const gradFinal = gradLogProb(tree.ref(q));
  p = addScaledGrad(p, gradFinal, halfStep);

  return [q, p];
}

/**
 * Single leapfrog step (for step-size initialization heuristic).
 *
 * @param position - Current position
 * @param momentum - Current momentum
 * @param gradLogProb - Gradient of log probability function
 * @param stepSize - Step size
 * @param massMatrix - Optional diagonal mass matrix
 * @returns [newPosition, newMomentum]
 */
export function leapfrogStep(
  position: JsTree<Array>,
  momentum: JsTree<Array>,
  gradLogProb: GradLogProbFn,
  stepSize: number,
  massMatrix?: JsTree<Array>,
): [JsTree<Array>, JsTree<Array>] {
  return leapfrog(position, momentum, gradLogProb, stepSize, 1, massMatrix);
}
