# Architectural Assessment: PR #2 vs PR #4

**Date:** 2026-01-17
**Reviewers:** Software Architect (Claude), API Compliance Reviewer

## Summary Metrics

| Aspect | PR #2 (hmc-impl) | PR #4 (jax-js-hmc-lib) |
|--------|------------------|------------------------|
| Lines added | 3,003 | 4,243 |
| Changed files | 32 | 22 |
| Test files | 9 | 7 |
| Commits | 4 | 8 |
| Includes browser example | ✅ | ❌ |
| Memory management docs | ❌ | ✅ |
| Reference tests scaffold | ✅ | ❌ |
| **API compliance with DESIGN.md** | ✅ | ❌ |

---

## Critical Issue: PR #4 Breaks the Documented API

PR #4 diverges from `DESIGN.md` in several user-visible ways that break documented usage patterns:

### 1. Flattened Draws (P2)

**Location:** `src/hmc.ts:295-305`

PR #4 flattens each sample via `flattenTree` and stacks flat vectors, so `draws` becomes `[numChains, numSamples, paramDim]` rather than a tree mirroring `initialParams`.

```typescript
// DESIGN.md expects this to work:
result.draws.mu    // ❌ Fails - draws is a flat array, not a tree
result.draws.x     // ❌ Fails
```

Any code following DESIGN.md will break because parameter structure is lost.

### 2. Diagnostics Type Mismatch (P2)

**Location:** `src/diagnostics.ts:151-156`

Diagnostics functions accept `number[][]`/`number[][][]` and use JS array methods, but `hmc` returns `np.Array`. Calling `summary(result.draws)` as documented will throw because `np.Array` doesn't have `.length`, `.map`, etc.

### 3. Non-Scalar Stats (P2)

**Location:** `src/hmc.ts:110-118`

Returns `acceptRate` and `stepSize` as `np.Array` (per-chain values) instead of scalar means as specified in DESIGN.md. Code that treats these as numbers will break when `numChains > 1`.

---

## PR #4: Better Internal Quality, Broken External API

### Strengths (Internal)

1. **More thorough physics tests** — 2-3x more comprehensive. `energy-conservation.test.ts` is 142 lines vs 40 lines in PR #2, testing pytree parameters and quadratic scaling.

2. **Better separation of concerns** — Types in `types.ts`, extracted `updatePosition`/`updateMomentum` helpers.

3. **Memory management documentation** — `JAX-JS-MEMORY.md` (175 lines) documents move semantics critical for maintainability.

4. **Cleaner diagnostics algorithms** — Pure JavaScript with clear Geyer's monotone sequence implementation.

5. **Disciplined commit history** — 8 commits showing TDD progression.

### Weaknesses (External)

- **API incompatible with DESIGN.md** — The three issues above make the documented examples non-functional
- No browser example
- No reference comparison tests

---

## PR #2: Rougher Internals, Correct API

### Strengths

1. **API compliance** — Returns draws as parameter tree, scalar stats, diagnostics work with output
2. **Browser example** — Working Vite example with WebGPU
3. **Reference test scaffold** — `tests/reference/` structure for NumPyro comparison
4. **Tree utilities** — `stackTrees`, `treeClone`, `treeDispose` preserve parameter structure

### Weaknesses

1. **Thinner tests** — Physics tests have minimal assertions
2. **Missing documentation** — No jax-js memory model explanation
3. **Monolithic HMC file** — 239 lines mixing concerns
4. **Interleaved adaptation** — Step size/mass matrix logic mixed with sampling

---

## Revised Verdict

The two PRs optimize for different things:

| Dimension | PR #2 | PR #4 |
|-----------|-------|-------|
| API compliance | ✅ Strong | ❌ Broken |
| Test thoroughness | ⚠️ Thin | ✅ Comprehensive |
| Documentation | ⚠️ Missing | ✅ Good |
| Code organization | ⚠️ Monolithic | ✅ Clean |

**For a library, API compliance is non-negotiable.** Users cannot work around a broken public interface, but they can live with imperfect internals.

### Recommendation

**Merge PR #2** — it delivers a working, spec-compliant library.

Then file issues to adopt PR #4's strengths:
1. Port `JAX-JS-MEMORY.md` documentation
2. Expand physics tests with PR #4's thoroughness
3. Extract helpers for better separation of concerns

Alternatively, if time permits: fix PR #4's API issues (preserve tree structure, align diagnostics types, return scalar stats) before merging. But that's effectively a rewrite of the public interface.
