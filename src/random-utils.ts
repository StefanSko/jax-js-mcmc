import { numpy as np, random, tree, Array, type JsTree } from "@jax-js/jax";

export function splitKeys(key: Array, num: number): Array[] {
  const keys = random.split(key, num);
  return np.split(keys, num, 0).map((k) => k.reshape([2]));
}

export function sampleNormalTree<T extends JsTree<Array>>(
  key: Array,
  template: T,
): { nextKey: Array; sample: T } {
  const leaves = tree.leaves(template) as Array[];
  const keys = splitKeys(key, leaves.length + 1);
  const nextKey = keys[0];
  const leafKeys = keys.slice(1);
  const samples = leaves.map((leaf, i) =>
    random.normal(leafKeys[i], leaf.shape),
  );
  const treedef = tree.structure(template);
  return { nextKey, sample: tree.unflatten(treedef, samples) as T };
}
