# Workflow for Jax-js-mcmc

A vibe-engineered workflow using multiple agents through design, implementation, and review phases.

---

## Design Phase

### Brainstorm with Superpowers (Claude Code)

- 2 worktrees via conductor

### Claude Code #1
- Plan mode
- Claude Code went off rails with respect to memory and refs
- Stopped and let it write a document about the issues, then restarted

### Claude Code #2
- Started to check it with the new memory doc and asked it to check `/tmp/blackjax`
- Let it do the plan, then asked it again to check on `/tmp/blackjax` when it got stuck more explicitly
- It got stuck, asked it to re-run, told it to add the deps to `.gitignore`

### Codex #2.1
- Started Codex to review and fix found issues
- ...

### Claude Code #2.2
- Ran code-simplifier
- Do PR #4

### Codex #3
- Let it run to check the plan from the brainstorming
- Clarifying questions
- Asked it to take suggested defaults and update the concrete design doc with that
- Told it to check `/tmp/jax-js` and check it before implementing
- It went off to create the thing
- Asked it to create a `Readme.md`
- Asked it to clone blackjax into `/tmp` for getting inspiration for more tests
- Asked it to cleanup PRs

### Codex #3.1
- Ran Codex `/review`, then fixed the found issues
- Do PR #2

---

## Review Phase

### Claude Code #4
- Compare both PRs
- Mentioned "Claude Code #2" as winner (Claude Code found Claude Code to be better)

### Codex #5
- Found "Codex #3" to be better
- "Claude Code #2" had breaking changes, uses arrays instead of trees
- Not reflecting the design doc

### Claude Code #4
- Input from Codex #5
- Agreed, but mentioned better testing suite from "Claude Code #2"
- Suggested: take "Codex #3" and cherry-pick some improvements from "Claude Code #2"

### Codex #5
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

### Claude Code #6
- Ran a code-simplifier
- Adapted PR

### Codex #5
- `/review` no regrets, push and merge
