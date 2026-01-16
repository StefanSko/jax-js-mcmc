import { numpy as np, type Array as JaxArray } from "@jax-js/jax";

import { treeMul, treeSum, type PyTree } from "./tree";

export function kineticEnergy(momentum: PyTree, inverseMassMatrix: PyTree): JaxArray {
  const scaled = treeMul(treeMul(momentum, momentum), inverseMassMatrix);
  return np.multiply(0.5, treeSum(scaled));
}

export function potentialEnergy(logProb: JaxArray): JaxArray {
  return np.negative(logProb);
}
