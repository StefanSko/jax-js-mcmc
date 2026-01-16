import { numpy as np, tree, type Array as JaxArray, type JsTree } from "@jax-js/jax";

export type PyTree = JsTree<JaxArray>;

export function treeAdd<T extends PyTree>(a: T, b: T): T {
  const aRef = tree.ref(a);
  const bRef = tree.ref(b);
  return tree.map((x, y) => np.add(x, y), aRef, bRef) as T;
}

export function treeSub<T extends PyTree>(a: T, b: T): T {
  const aRef = tree.ref(a);
  const bRef = tree.ref(b);
  return tree.map((x, y) => np.subtract(x, y), aRef, bRef) as T;
}

export function treeMul<T extends PyTree>(a: T, b: T): T {
  const aRef = tree.ref(a);
  const bRef = tree.ref(b);
  return tree.map((x, y) => np.multiply(x, y), aRef, bRef) as T;
}

export function treeScale<T extends PyTree>(a: T, scalar: number): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.multiply(x, scalar), aRef) as T;
}

export function treeDivScalar<T extends PyTree>(a: T, scalar: number): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.divide(x, scalar), aRef) as T;
}

export function treeAddScalar<T extends PyTree>(a: T, scalar: number): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.add(x, scalar), aRef) as T;
}

export function treeNegate<T extends PyTree>(a: T): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.negative(x), aRef) as T;
}

export function treeReciprocal<T extends PyTree>(a: T): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.reciprocal(x), aRef) as T;
}

export function treeSqrt<T extends PyTree>(a: T): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.sqrt(x), aRef) as T;
}

export function treeZerosLike<T extends PyTree>(a: T): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.zerosLike(x), aRef) as T;
}

export function treeOnesLike<T extends PyTree>(a: T): T {
  const aRef = tree.ref(a);
  return tree.map((x) => np.onesLike(x), aRef) as T;
}

export function treeSum(a: PyTree): JaxArray {
  let total = np.array(0);
  for (const leaf of tree.leaves(tree.ref(a))) {
    total = np.add(total, np.sum(leaf));
  }
  return total;
}

export function treeDot(a: PyTree, b: PyTree): JaxArray {
  const leavesA = tree.leaves(tree.ref(a));
  const leavesB = tree.leaves(tree.ref(b));
  if (leavesA.length !== leavesB.length) {
    throw new Error("treeDot requires trees with the same number of leaves");
  }
  let total = np.array(0);
  for (let i = 0; i < leavesA.length; i += 1) {
    total = np.add(total, np.sum(np.multiply(leavesA[i], leavesB[i])));
  }
  return total;
}

export function stackTrees<T extends PyTree>(trees: T[], axis = 0): T {
  if (trees.length === 0) {
    throw new Error("stackTrees requires at least one tree");
  }
  const refs = trees.map((entry) => tree.ref(entry));
  return tree.map((...xs: JaxArray[]) => np.stack(xs, axis), refs[0], ...refs.slice(1)) as T;
}
