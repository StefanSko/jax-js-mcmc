import { describe, expect, test } from "vitest";

import { random, type Array as JaxArray } from "@jax-js/jax";

import { findReasonableStepSize } from "../../src/adaptation";
import { splitKey } from "../../src/random";

describe("dual averaging step size", () => {
  test("returns finite step size and responds to target accept rate", () => {
    const evaluator = (key: JaxArray, stepSize: number) => ({
      acceptProb: 1 / (1 + stepSize),
      nextKey: key,
    });

    const [keyA, keyB] = splitKey(random.key(42), 2);
    const highTarget = findReasonableStepSize(keyA, evaluator, 0.9, 0.1);
    const lowTarget = findReasonableStepSize(keyB, evaluator, 0.1, 0.1);

    expect(Number.isFinite(highTarget.stepSize)).toBe(true);
    expect(Number.isFinite(lowTarget.stepSize)).toBe(true);
    expect(highTarget.stepSize).toBeLessThan(lowTarget.stepSize);
  });
});
