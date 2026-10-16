# Contributing

How maintainers work on the library. If you are writing an experiment in your own
fork, you don't need this file: [CLAUDE.md](CLAUDE.md) has the commands and
conventions.

## Goal

Keep the code easy to read, change and extend, so the library can scale. More
advanced constructs are fine when they make things clearer at the point of use,
e.g. grouping related arguments (`f(x, x_bounds)` instead of
`f(x, x_lower, x_upper)`). Avoid complexity that doesn't serve that, such as
abstractions with a single use or boilerplate from static languages.

Planned work and known bugs are GitHub issues. The design and the reasons for it
are in [DESIGN.md](DESIGN.md), and the decisions that are hard to reverse in
[docs/adr/](docs/adr/). Read them before changing the structure, and update
DESIGN.md when a PR changes the design.

## Workflow

- `main` is the working branch. One PR per issue into `main`. Branch from it using
  a short topic name (e.g. `gp-priors`).
- Start a ticket with `/mattpocock-skills:implement #<n>`. Otherwise, agree scope
  in plan mode before editing.
- Before opening a PR, run `pixi run -e dev check` and `pixi run -e dev test`,
  then `/mattpocock-skills:code-review` since `main`.
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
- A PR that changes usage, the public API or behavior updates the README and the
  user docs (`docs/*.rst`), and adds a line to [CHANGELOG.md](CHANGELOG.md).
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
