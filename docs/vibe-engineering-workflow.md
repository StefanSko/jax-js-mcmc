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
- Ran `/review` against base branch `main`
- Reviewed memory/ref handling in Welford updates
- Found potential issues with slice function usage on 2D arrays
- Analyzed acceptance probability handling and step size adaptation

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

---

## Transcripts

Published session transcripts for reference.

### Claude Code Sessions

| Session | Description | Transcript |
|---------|-------------|------------|
| Claude Code #1 | Memory doc session (Lyon) | [gist](https://gisthost.github.io/?4236cd477baba3bbc7e4986d311d0da4/index.html) |
| Claude Code #1 subagent | Subagent for #1 | [gist](https://gisthost.github.io/?59c83b619e8bbfa8c7db6a8c111343df/index.html) |
| Claude Code #2 | Plan mode, blackjax investigation (Kyoto) | [gist](https://gisthost.github.io/?07d34dfbff597a129485d4da89ed1211/index.html) |
| Claude Code #2.2 | Code-simplifier, PR #4 prep (Kyoto) | [gist](https://gisthost.github.io/?75d65e8029cc900b6c2e66131c0379ac/index.html) |
| Claude Code #4 | Comparing PRs (Cairo) | [gist](https://gisthost.github.io/?2252cde61718e48777f05579c123ab5b/index.html) |
| Claude Code #6 | Code-simplifier on PR #4 (Kyoto) | [gist](https://gisthost.github.io/?e36e8fccd9809e35748750aac804bfed/index.html) |
| Claude Code #6 subagent | Subagent for #6 | [gist](https://gisthost.github.io/?5fa2e55bc20fab49e6988631f0776f82/index.html) |
| Later work | PR #6 enhancements | [gist](https://gisthost.github.io/?d20590487339a9cc0937e08bc40a2457/index.html) |
| Kyoto subagent a8dfbd5 | Additional subagent | [gist](https://gisthost.github.io/?9d640cd31509a1d2925d262d44b16f8c/index.html) |
| Kyoto subagent a9b447c | Additional subagent | [gist](https://gisthost.github.io/?193576e0f9de7a84d450d90fa064ae8d/index.html) |
| Kyoto subagent a4c9523 | Additional subagent | [gist](https://gisthost.github.io/?e7310786e205edb930a98cf3f696b0a6/index.html) |

### Codex Sessions

| Session | Description | Transcript |
|---------|-------------|------------|
| Codex #2.1 | Review and fix issues | [gist](https://gistpreview.github.io/?22deee051c7926f93037cd20fc70da99/index.html) |
| Codex #3 | Implementation session | [gist](https://gistpreview.github.io/?5f6b8022e4572a7a886db872d7df98fb/index.html) |
| Codex #3.1 (a) | /review and PR #2 | [gist](https://gistpreview.github.io/?d039f47a6bb9d9a361a65cd6aa1204a0/index.html) |
| Codex #3.1 (b) | PR #2 finalization | [gist](https://gistpreview.github.io/?98c9907b74907a71996874661dc425fe/index.html) |
| Codex #5 (a) | Review and comparison | [gist](https://gistpreview.github.io/?fcd729f7e831fc0f70021e3e6c4c835c/index.html) |
| Codex #5 (b) | Cherry-picking, /review, merge | [gist](https://gistpreview.github.io/?96f339ef48f2c1fbd33830169e129423/index.html) |
