export function acceptanceProbability(deltaHamiltonian: number): number {
  if (!Number.isFinite(deltaHamiltonian)) {
    return 0;
  }
  if (deltaHamiltonian <= 0) {
    return 1;
  }
  const prob = Math.exp(-deltaHamiltonian);
  return prob >= 1 ? 1 : prob;
}
