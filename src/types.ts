import type { numpy as np } from "@jax-js/jax";

/**
 * A JsTree is a nested structure of arrays, objects, or arrays of arrays.
 * Matches the jax-js tree utility conventions.
 */
export type JsTree<T> = T | JsTree<T>[] | { [key: string]: JsTree<T> };

/**
 * Log probability function type.
 * Must return a scalar Array (0-dimensional).
 */
export type LogProbFn<T extends JsTree<np.Array>> = (params: T) => np.Array;

/**
 * Gradient of log probability function type.
 * Returns the same tree structure as the input.
 */
export type GradLogProbFn<T extends JsTree<np.Array>> = (params: T) => T;
