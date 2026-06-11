# docs/convos/ — session and conversation artifacts

Process metadata only: plans, session summaries, and decision logs. **Not** research
deliverables — those live in `docs/notes/`.

## What goes here

- Plans that drove a session (typically a copy of a `~/.claude/plans/*.md` once approved)
- Session summaries: what ran, what changed, what was decided, with commit hashes
- Standalone decision logs that aren't tied to a single research artifact

## What does NOT go here

- Briefs, supplementary notes, handoff docs, results write-ups → `docs/notes/`
- Long-form planning that maps to a project phase → `docs/<phase>_plan.md` (next to the
  existing `docs/journal_hardening_plan.md`, `docs/conference_resubmission_plan.md`)

## Naming

`YYYY-MM-DD_<slug>_<kind>.md` — date sorts chronologically, slug groups by topic, kind
disambiguates plan vs summary vs decision-log.

Examples:

```
2026-05-28_mina-closeout_plan.md
2026-05-28_mina-closeout_summary.md
```

If a future session has multiple plans or summaries for the same topic, suffix with
`_v2`, `_followup`, etc.

## Why separate from docs/notes/

Research docs need to stay clean for the manuscript: a reviewer or co-author opening
`docs/notes/` should see briefs and supplementary findings, not "what Claude did on
2026-05-28". Conversely, session logs reference commits and process detail that would
just clutter a research note.
