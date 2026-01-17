# Cherry-Pick Plan: PR #4 Improvements

**Date:** 2026-01-17
**Status:** Ready for implementation
**Source:** PR #4 (StefanSko/jax-js-hmc-lib)
**Target:** main (after PR #2 merge)

## Overview

PR #4 has valuable improvements that should be adopted without breaking the API established by PR #2. This plan identifies what to cherry-pick and how.

---

## 1. JAX-JS Memory Management Guide

**Priority:** High
**Effort:** Direct cherry-pick
**Commit:** `23ee937`

### What
The `docs/JAX-JS-MEMORY.md` file (175 lines) documenting jax-js move semantics and `.ref` patterns.

### Why
Critical knowledge for anyone maintaining or extending this codebase. The move semantics of jax-js are non-obvious and this documentation prevents bugs.

### How
```bash
git cherry-pick 23ee937
```

This commit only adds a new file with no conflicts expected.

### Follow-up
Update `CLAUDE.md` to reference the memory guide in the prerequisites section.

---

## 2. Enhanced Physics Tests

**Priority:** High
**Effort:** Manual adaptation required
**Source files:** `tests/physics/*.test.ts` from PR #4

### What
PR #4's physics tests are 2-3x more comprehensive:
- `energy-conservation.test.ts`: 142 lines vs 40 lines
- `reversibility.test.ts`: 183 lines vs 27 lines
- `volume-preservation.test.ts`: 184 lines vs 54 lines
- `detailed-balance.test.ts`: 252 lines vs 87 lines

### Why
More thorough tests catch edge cases and document expected physics behavior.

### How
Cannot directly cherry-pick because:
1. PR #4 uses `tree.ref()` vs PR #2's `treeRef()` helper
2. Import paths differ
3. Some tests depend on PR #4's flattened structure

**Approach:** Manual port
1. Create branch `improve-physics-tests`
2. For each test file, copy test cases from PR #4
3. Adapt imports to use PR #2's tree utilities
4. Adapt assertions to work with PR #2's tree-based draws
5. Run tests to verify

### Specific additions to port

#### energy-conservation.test.ts
- [ ] Add "energy is conserved for small step sizes" test
- [ ] Add "energy conservation with pytree parameters" test

#### reversibility.test.ts
- [ ] Add "reversibility with pytree parameters" test
- [ ] Add "reversibility with mass matrix" test

#### volume-preservation.test.ts
- [ ] Add numerical Jacobian computation helper
- [ ] Add "volume preservation with different step sizes" test

#### detailed-balance.test.ts
- [ ] Add "acceptance probability histogram" test
- [ ] Add statistical validation of Metropolis rule

---

## 3. ESS Diagnostics Tests

**Priority:** Medium
**Effort:** Manual adaptation
**Commit:** `a649b36`

### What
Test file `tests/diagnostics/ess.test.ts` validating ESS computation.

### Why
PR #2's diagnostics are untested for ESS edge cases.

### How
1. Create `tests/diagnostics/` directory (PR #2 has flat `tests/diagnostics.test.ts`)
2. Port test cases, adapting for PR #2's `np.Array`-based diagnostics

---

## 4. Code Organization Improvements

**Priority:** Low
**Effort:** Refactoring PR

### What
PR #4 has cleaner separation:
- `types.ts` for type definitions
- Extracted `updatePosition`/`updateMomentum` helpers in leapfrog

### Why
Improves maintainability and readability.

### How
Create separate refactoring PR:
1. Extract types to `src/types.ts`
2. Extract leapfrog helpers
3. Ensure all tests still pass

**Note:** This is purely internal refactoring with no API changes.

---

## Implementation Order

| Phase | Task | Blocked By |
|-------|------|------------|
| 1 | Cherry-pick memory guide | - |
| 2 | Port enhanced physics tests | Phase 1 |
| 3 | Port ESS diagnostics tests | - |
| 4 | Refactor code organization | Phases 2-3 |

---

## Commands

```bash
# Phase 1: Memory guide
git checkout -b docs/memory-guide
git cherry-pick 23ee937
# Update CLAUDE.md to reference the guide
git push -u origin docs/memory-guide
gh pr create --title "Add jax-js memory management guide" --body "Cherry-picked from PR #4"

# Phase 2: Enhanced tests (manual work)
git checkout main
git checkout -b improve-physics-tests
# ... manual porting work ...

# Phase 3: ESS tests
git checkout main
git checkout -b add-ess-tests
# ... manual porting work ...

# Phase 4: Refactoring
git checkout main
git checkout -b refactor/code-organization
# ... refactoring work ...
```

---

## What NOT to Cherry-Pick

| Commit | Reason |
|--------|--------|
| `8126510` (HMC implementation) | Incompatible API (flattened draws) |
| `53c3704` (helper extraction) | Depends on incompatible structure |
| `47f5d11` (ESS fix) | May not apply to PR #2's ESS impl |
| `cafc5ac` (revert) | Meta-commit, not useful |

---

## Success Criteria

- [ ] `docs/JAX-JS-MEMORY.md` exists and is referenced in CLAUDE.md
- [ ] Physics tests cover pytree parameters
- [ ] Physics tests verify quadratic energy scaling
- [ ] ESS has dedicated test coverage
- [ ] All existing tests continue to pass
