import type { Array, JsTree } from "@jax-js/jax";
import { mapTree } from "./tree-utils";

export function leapfrog<Params extends JsTree<Array>>(
  position: Params,
  momentum: Params,
  gradLogProb: (p: Params) => Params,
  stepSize: number,
  numSteps: number,
  massMatrix: Params,
): [Params, Params] {
  let q = position;
  let p = momentum;

  let grad = gradLogProb(q);
  p = mapTree((pi: Array, gi: Array) => pi.add(gi.mul(stepSize / 2)), p, grad) as Params;

  for (let i = 0; i < numSteps; i++) {
    q = mapTree(
      (qi: Array, pi: Array, mi: Array) => qi.add(pi.div(mi).mul(stepSize)),
      q,
      p,
      massMatrix,
    ) as Params;
    grad = gradLogProb(q);
    if (i !== numSteps - 1) {
      p = mapTree((pi: Array, gi: Array) => pi.add(gi.mul(stepSize)), p, grad) as Params;
    }
  }

  p = mapTree((pi: Array, gi: Array) => pi.add(gi.mul(stepSize / 2)), p, grad) as Params;
  return [q, p];
}
