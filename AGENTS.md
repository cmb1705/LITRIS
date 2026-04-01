# Agent Instructions

## Beads Shared Server Policy

This repo uses `bd` on the canonical local shared Dolt server. Run `bd prime`
at session start to recover workflow context.

Canonical shared-server root:

- `C:\Users\cmb17\.beads\shared-server`
- `127.0.0.1:3308`
- database name: `LITRIS`
- issue prefix: `LITRIS`

Use `bd dolt show` or `bd context` if you need to verify the active server and
database wiring. Do not create alternate repo-local beads stores, umbrella
trackers for parent folders, or ad hoc replacement databases when configuration
drifts. Fix the existing shared-server configuration instead.

There is no remote Dolt sync for beads in this workspace. Do not run
`bd dolt push` or `bd dolt pull`.

## Quick Reference

```bash
bd prime
bd ready
bd show <id>
bd update <id> --claim
bd close <id>
bd sync
bd remember "persistent insight"
```

## Rules

- Use `bd` for all task tracking.
- Do not use `TodoWrite`, `TaskCreate`, markdown TODOs, or other side trackers.
- Use `bd remember` for persistent knowledge. Do not use `MEMORY.md` files.

## Session Close

When ending a session, update issue status, run any required quality checks,
commit code changes, and push git changes. `bd` state persists locally on the
shared server; use `bd sync` only for repo-side JSONL export when needed.
