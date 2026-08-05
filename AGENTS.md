# Repository Guidelines

## Sources of Truth

Projecta is documentation-first. Begin with [`README.md`](./docs/README.md), then read the initialization documents relevant to the task. Use this precedence when instructions conflict:

1. Current user instruction.
2. `AGENTS.md`.
3. `.agents/memory/decisions.md`.
4. `docs/`.
5. Applicable `.agents/rules/`.
6. Existing implementation.

Read `.agents/memory/decisions.md` before architecture or technology changes. Use `$log-decision` or `/log-decision` for an approved decision; do not rewrite prior entries. Before diagnosing a recurring bug, search `.agents/memory/lessons-learned.md` when it exists. Use `$log-lesson` or `/log-lesson` after resolving a reusable bug or environment-specific issue.

## Project Structure

You can use `$graphify` or `/graphify` to understand codebase.

For document structure, see [`README.md`](./docs/README.md).

For agent guideline, see [`.agents`](./.agents) folder.

## Rules and Skills

Load the applicable source rather than duplicating it:

- Git and commits: `.agents/rules/git.md`
- Markdown: `.agents/rules/markdown.md`
- Python: `.agents/rules/python.md`
- Changelog: `.agents/rules/changelog.md`
- Shell / Powershell execution: `.agents/rules/shell.md`

## Build and Validation

No executable application or root build configuration exists yet. Use:

```text
rg --files docs .agents
git diff --check
```

## Testing and Review

PRs should state intent, affected boundaries, validation evidence, and unresolved risks. Reference the applicable rules instead of restating them. Never commit secrets, credentials, sensitive payloads, or production data.

## Graphify
You should use graphify to understand codebase more effiency, instead of scan all codebase. See `$graphify` skills