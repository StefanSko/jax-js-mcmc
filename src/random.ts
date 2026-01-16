import { numpy as np, random, type Array as JaxArray } from "@jax-js/jax";

export function splitKey(key: JaxArray, num = 2): JaxArray[] {
  const keys = random.split(key, num);
  const parts = np.split(keys, num, 0);
  return parts.map((part) => np.squeeze(part));
}
