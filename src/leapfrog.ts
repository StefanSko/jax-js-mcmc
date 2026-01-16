import { tree } from "@jax-js/jax";

import { treeAdd, treeMul, treeOnesLike, treeScale, type PyTree } from "./tree";

export function leapfrog(
  position: PyTree,
  momentum: PyTree,
  gradPotential: (q: PyTree) => PyTree,
  stepSize: number,
  numSteps: number,
  inverseMassMatrix?: PyTree,
): [PyTree, PyTree] {
  if (numSteps <= 0 || stepSize === 0) {
    return [position, momentum];
  }

  let q = position;
  let p = momentum;
  const invMass = inverseMassMatrix ?? treeOnesLike(position);

  let grad = gradPotential(tree.ref(q) as PyTree);
  p = treeAdd(p, treeScale(grad, -0.5 * stepSize));

  for (let step = 0; step < numSteps; step += 1) {
    const velocity = treeMul(invMass, p);
    q = treeAdd(q, treeScale(velocity, stepSize));

    grad = gradPotential(tree.ref(q) as PyTree);
    const scale = step === numSteps - 1 ? -0.5 * stepSize : -1.0 * stepSize;
    p = treeAdd(p, treeScale(grad, scale));
  }

  return [q, p];
}
