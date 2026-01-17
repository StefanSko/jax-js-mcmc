import { beforeAll, describe, expect, test } from "vitest";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { kineticEnergy } from "../../src/hamiltonian";
import { treeRef } from "../../src/tree-utils";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

describe("kinetic energy", () => {
  test("matches 0.5 * sum(p^2 / m)", () => {
    const momentum = np.array([2.0, 3.0]);
    const massMatrix = np.array([4.0, 9.0]);

    const energy = kineticEnergy(momentum, massMatrix).item();
    expect(Math.abs(energy - 1.0)).toBeLessThan(1e-6);
  });

  test("matches sum across pytree leaves", () => {
    const momentum = {
      x: np.array([1.0, -2.0]),
      y: np.array([3.0]),
    };
    const massMatrix = {
      x: np.array([2.0, 4.0]),
      y: np.array([9.0]),
    };

    const energy = kineticEnergy(
      treeRef(momentum),
      treeRef(massMatrix),
    ).item();

    const expected = 0.5 * ((1 / 2 + 4 / 4) + 9 / 9);
    expect(Math.abs(energy - expected)).toBeLessThan(1e-6);
  });
});
