# Contributing

How maintainers work on the library during the refactor. If you are writing an
experiment in your own fork, you don't need this file: [CLAUDE.md](CLAUDE.md) has the
commands and conventions.

## Goal of the refactor

Make the code easier to read, change and extend, so the library can scale. More
advanced constructs are fine when they make things clearer at the point of use,
e.g. grouping related arguments (`f(x, x_bounds)` instead of
`f(x, x_lower, x_upper)`). Avoid complexity that doesn't serve that, such as
abstractions with a single use or boilerplate from static languages.

The roadmap is in [ROADMAP.md](ROADMAP.md). Check it before starting work, and
tick items off in the PR that completes them. The target API and the plan for
each restructure PR are in [DESIGN.md](DESIGN.md), and the decisions behind it in
[docs/adr/](docs/adr/). Read them before any restructure PR, and update DESIGN.md
when a PR changes the design.

## Workflow

- `refactor/initial` is the working branch. `main` stays frozen until the roadmap
  is complete. Never commit to or open PRs against `main`. Until then
  `refactor/initial` is also the GitHub default branch, because closing keywords
  (`Closes #<n>`) only close issues on merges into the default branch.
- One PR per issue into `refactor/initial`. Branch from it using the name
  `<stage>-<topic>` (e.g. `s1-hooks`).
- Start a ticket with `/mattpocock-skills:implement #<n>`. Otherwise, agree scope
  in plan mode before editing.
- Before opening a PR, run `pixi run -e dev check` and `pixi run -e dev test`,
  then `/mattpocock-skills:code-review` since `refactor/initial`.
- Write the PR body with the `pr` skill (`mattpocock-skills:pr`; if it isn't
  installed, follow `.github/pull_request_template.md`, which has its sections). End
  it with `Closes #<n>`, which is how code-review finds the spec. Add the
  `ai-assisted` label to PRs with AI-written code. Commits reference their issue
  (`#<n>`) and get a `Co-Authored-By` trailer.
- Keep non-logic and logic changes in separate PRs. A *logic* PR changes the
  results of existing code; adding new API that leaves existing behavior
  unchanged is not logic.
- A PR that updates the regression baselines explains the JSON diff.
- A PR that changes behavior or contracts updates the relevant skill
  (`.claude/skills/`).
- Keep docs concise. Sphinx autoapi already covers the API, so don't duplicate it.
- When a style question comes up, ask, then record the answer under Conventions in
  CLAUDE.md.

## Agent skills

### Issue tracker

GitHub Issues on `jj2249/hybrid-bo` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.
