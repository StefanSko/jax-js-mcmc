import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, Array, tree } from "@jax-js/jax";
import { leapfrog } from "../../src/leapfrog.js";

describe("debug ref counting", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("basic ref counting - correct usage", () => {
    const x = np.array([1, 2, 3], { dtype: np.DType.Float32 });

    // To use x multiple times, take .ref before each use
    // x.ref returns x with incremented refcount, but it's still the same object
    const result1 = x.ref.mul(2);  // This increments then decrements ref
    expect(result1.js()).toEqual([2, 4, 6]);

    const result2 = x.ref.mul(3);  // Same - increment then decrement
    expect(result2.js()).toEqual([3, 6, 9]);

    // Final use without .ref consumes x
    const result3 = x.mul(4);
    expect(result3.js()).toEqual([4, 8, 12]);

    // Now x is consumed, can't use it anymore
    // expect(() => x.js()).toThrow(); // This would throw
  });

  test("leapfrog with simple gradient", () => {
    function gradLogProb(q: Array): Array {
      return q.mul(-1) as Array;
    }

    const q0 = np.array([1.0, 0.5], { dtype: np.DType.Float32 });
    const p0 = np.array([0.0, 1.0], { dtype: np.DType.Float32 });

    const [q1, p1] = leapfrog(q0, p0, gradLogProb, 0.1, 10);

    // Should be able to access results
    expect((q1 as Array).shape).toEqual([2]);
    expect((p1 as Array).shape).toEqual([2]);
  });

  test("function that preserves caller's array", () => {
    // The function uses x.ref internally so caller can reuse x
    function doubleAndReturn(x: Array): Array {
      return x.ref.mul(2) as Array;
      // x is consumed by this function, but because we used x.ref,
      // the caller's reference is preserved
    }

    let x = np.array([1, 2, 3], { dtype: np.DType.Float32 });

    // Each call consumes the passed array, so we pass x.ref
    const r1 = doubleAndReturn(x.ref);
    expect(r1.js()).toEqual([2, 4, 6]);

    const r2 = doubleAndReturn(x.ref);
    expect(r2.js()).toEqual([2, 4, 6]);

    // Final call without ref
    const r3 = doubleAndReturn(x);
    expect(r3.js()).toEqual([2, 4, 6]);
  });

  test("loop with ref", () => {
    let x = np.array([1, 2, 3], { dtype: np.DType.Float32 });

    // In a loop, use x.ref to pass to functions
    for (let i = 0; i < 3; i++) {
      // Create new x each iteration
      x = x.ref.mul(2) as Array;
    }

    expect(x.js()).toEqual([8, 16, 24]);
  });

  test("tree.map in loop", () => {
    let x = np.array([1, 2, 3], { dtype: np.DType.Float32 });

    // tree.map consumes the input trees
    for (let i = 0; i < 3; i++) {
      // Inside the map function, use .ref if you need to use leaf multiple times
      x = tree.map((a: Array) => a.ref.mul(2) as Array, x) as Array;
    }

    expect(x.js()).toEqual([8, 16, 24]);
  });
});
