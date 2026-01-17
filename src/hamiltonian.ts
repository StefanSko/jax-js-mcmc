import { Array, type JsTree } from "@jax-js/jax";
import { mapTree, treeClone, treeSum, treeRef } from "./tree-utils";

export function kineticEnergy(
  momentum: JsTree<Array>,
  massMatrix: JsTree<Array>,
): Array {
  const scaled = mapTree(
    (p: Array, m: Array) => p.mul(p.ref).div(m.ref),
    momentum,
    massMatrix,
  ) as JsTree<Array>;
  return treeSum(scaled).mul(0.5);
}

export function hamiltonian<Params extends JsTree<Array>>(
  position: Params,
  momentum: Params,
  logProb: (p: Params) => Array,
  massMatrix: Params,
): Array {
  const potential = logProb(treeClone(position)).mul(-1);
  const kinetic = kineticEnergy(treeRef(momentum), massMatrix);
  return potential.add(kinetic);
}
