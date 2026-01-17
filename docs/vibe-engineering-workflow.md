# Workflow for Jax-js-mcmc

A vibe-engineered workflow using multiple agents through design, implementation, and review phases.

---

## Design Phase

### Brainstorm with Superpowers (Claude Code)

- 2 worktrees via conductor

### [Claude Code #1](https://gisthost.github.io/?68f03be6c5d158c0276aa81356f82228/index.html)
- Plan mode
- Claude Code went off rails with respect to memory and refs
- Stopped and let it write a document about the issues, then restarted

### [Claude Code #2](https://gisthost.github.io/?07d34dfbff597a129485d4da89ed1211/index.html)
- Started to check it with the new memory doc and asked it to check `/tmp/blackjax`
- Let it do the plan, then asked it again to check on `/tmp/blackjax` when it got stuck more explicitly
- It got stuck, asked it to re-run, told it to add the deps to `.gitignore`

### [Codex #2.1](https://gistpreview.github.io/?22deee051c7926f93037cd20fc70da99/index.html)
- Ran `/review` against base branch `main`
- Reviewed memory/ref handling in Welford updates
- Found potential issues with slice function usage on 2D arrays
- Analyzed acceptance probability handling and step size adaptation

### [Claude Code #2.2](https://gisthost.github.io/?75d65e8029cc900b6c2e66131c0379ac/index.html)
- Explored jax-js API patterns and conventions
- Created implementation plan with TDD approach
- Started project setup

### [Codex #3](https://gistpreview.github.io/?5f6b8022e4572a7a886db872d7df98fb/index.html)
- Let it run to check the plan from the brainstorming
- Clarifying questions
- Asked it to take suggested defaults and update the concrete design doc with that
- Told it to check `/tmp/jax-js` and check it before implementing
- It went off to create the thing
- Asked it to create a `Readme.md`
- Asked it to clone blackjax into `/tmp` for getting inspiration for more tests
- Asked it to cleanup PRs

**API/Types notes from Codex #3:**
- `logProb(params)` returns a scalar Array (0-dim) in float32; no JS number return
- `initialParams` required; `key` required; `numSamples` required
- `JsTree` = nested object/array of Array leaves; use jax-js tree utilities for map/flatten
- HMC options with defaults

### [Codex #3.1](https://gistpreview.github.io/?d039f47a6bb9d9a361a65cd6aa1204a0/index.html)
- Ran Codex `/review`, then fixed the found issues
- Do PR #2

---

## Review Phase

### Claude Code #4
- Compare both PRs
- Mentioned "Claude Code #2" as winner (Claude Code found Claude Code to be better)

### [Codex #5](https://gistpreview.github.io/?fcd729f7e831fc0f70021e3e6c4c835c/index.html)
- Found "Codex #3" to be better
- "Claude Code #2" had breaking changes, uses arrays instead of trees
- Not reflecting the design doc

### Claude Code #4
- Input from Codex #5
- Agreed, but mentioned better testing suite from "Claude Code #2"
- Suggested: take "Codex #3" and cherry-pick some improvements from "Claude Code #2"

### [Codex #5](https://gistpreview.github.io/?96f339ef48f2c1fbd33830169e129423/index.html)
- Passed on new suggestion from Claude Code #4
- Mostly agreed, asked it to plan/implement
- Went off to add the improvements from "Claude Code #2"
- Realized dev dependency instructions around vitest not paid attention to
- `.agents/Agents.md` not in context
- Streamlined setup / added `pnpm install` to `Agents.md`
- Loosened volume preservation tests
- `random.split` → `splitKeys`
- Checked blackjax for inspiration, added:
  1. One analytic system test (harmonic oscillator)
  2. A kinetic-energy/mass-matrix math test
  3. A multi-step trajectory check (e.g., final state near analytic after N steps)
- Pushed commit and adapted PR

### [Claude Code #6](https://gisthost.github.io/?e36e8fccd9809e35748750aac804bfed/index.html)
- Ran a code-simplifier
- Adapted PR

### Codex #5
- `/review` no regrets, push and merge

---

## Transcripts

Published session transcripts for reference.

### Claude Code Sessions

| Session | Description | Transcript |
|---------|-------------|------------|
| Claude Code #1 | Plan mode, memory/refs issues, JAX-JS-MEMORY.md (Lyon) | [gist](https://gisthost.github.io/?68f03be6c5d158c0276aa81356f82228/index.html) |
| Claude Code #2 | HMC implementation, debugging memory/move semantics (Kyoto) | [gist](https://gisthost.github.io/?07d34dfbff597a129485d4da89ed1211/index.html) |
| Claude Code #2.2 | jax-js API exploration, implementation plan, project setup (Kyoto) | [gist](https://gisthost.github.io/?75d65e8029cc900b6c2e66131c0379ac/index.html) |
| Claude Code #6 | Code-simplifier on PR #4 (Kyoto) | [gist](https://gisthost.github.io/?e36e8fccd9809e35748750aac804bfed/index.html) |
| Later work | PR #6 enhancements, code-simplifier | [gist](https://gisthost.github.io/?d20590487339a9cc0937e08bc40a2457/index.html) |

### Codex Sessions

| Session | Description | Transcript |
|---------|-------------|------------|
| Codex #2.1 | Review ESS bugs, fix normalization and chain handling | [gist](https://gistpreview.github.io/?22deee051c7926f93037cd20fc70da99/index.html) |
| Codex #3 | Full HMC implementation with design review, tests, PR consolidation | [gist](https://gistpreview.github.io/?5f6b8022e4572a7a886db872d7df98fb/index.html) |
| Codex #3.1 (a) | Mass matrix timing bug fix, step-size retuning | [gist](https://gistpreview.github.io/?d039f47a6bb9d9a361a65cd6aa1204a0/index.html) |
| Codex #3.1 (b) | PR #2 code review, mass matrix adaptation analysis | [gist](https://gistpreview.github.io/?98c9907b74907a71996874661dc425fe/index.html) |
| Codex #5 (a) | TDD strategy, BlackJAX integration, implementation | [gist](https://gistpreview.github.io/?fcd729f7e831fc0f70021e3e6c4c835c/index.html) |
| Codex #5 (b) | PR comparison, cherry-picking, test expansion, merge | [gist](https://gistpreview.github.io/?96f339ef48f2c1fbd33830169e129423/index.html) |
