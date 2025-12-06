---
name: verification
description: Systematic verification checklist for task completion. Use before marking any task, issue, or phase as complete.
---

# Task Completion Verification Skill

This skill provides systematic verification checklists to ensure tasks are properly completed before marking them done.

## When to Use

Invoke this skill before:
- Marking a task file checkbox as complete `[ ]` -> `[x]`
- Updating STATUS in task files to "Complete"
- Closing issues in ISSUES.md
- Updating phase completion in STATE.md

## Verification Checklists

### Code Implementation Tasks

Before marking code tasks complete, verify:

1. **Specification Compliance**
   - [ ] Implementation matches the specification document
   - [ ] All acceptance criteria from the task file are met
   - [ ] No functionality is missing or stubbed

2. **Integration**
   - [ ] All imported functions are actually used
   - [ ] Validators are called where specified
   - [ ] Error handling returns structured responses (not raw exceptions)

3. **Testing**
   - [ ] Unit tests exist and pass
   - [ ] Integration tests cover the feature
   - [ ] Error paths are tested

4. **Documentation**
   - [ ] Code has appropriate docstrings
   - [ ] README/docs updated if needed
   - [ ] All documentation links resolve

### Bug Fix Tasks

Before marking bug fixes complete, verify:

1. **Root Cause**
   - [ ] Root cause identified and documented
   - [ ] Fix addresses the root cause (not just symptoms)

2. **Testing**
   - [ ] Regression test added
   - [ ] Related functionality still works
   - [ ] Edge cases tested

3. **No Regressions**
   - [ ] All existing tests pass
   - [ ] No new warnings or errors

### Documentation Tasks

Before marking documentation tasks complete, verify:

1. **Content**
   - [ ] Information is accurate and current
   - [ ] All code examples work
   - [ ] All links resolve

2. **Completeness**
   - [ ] All promised sections exist
   - [ ] Examples cover common use cases
   - [ ] Troubleshooting covers known issues

### Phase/Milestone Completion

Before marking phases complete, verify:

1. **All Tasks**
   - [ ] Every task in the phase is complete
   - [ ] Each task was individually verified

2. **Integration**
   - [ ] All components work together
   - [ ] End-to-end tests pass

3. **Documentation**
   - [ ] Phase documentation is complete
   - [ ] STATE.md reflects accurate status

## Verification Commands

```bash
# Check for unused imports
ruff check --select F401 src/

# Run all tests
pytest tests/ -v

# Check documentation links
grep -ohE '\[.*\]\([^)]+\.md\)' *.md docs/*.md | grep -oE '\([^)]+\)' | tr -d '()' | while read link; do
  if [ ! -f "$link" ]; then echo "Broken: $link"; fi
done

# Verify validators are used
for v in $(grep "def validate_" src/mcp/validators.py | sed 's/def \(.*\)(.*/\1/'); do
  if ! grep -rq "$v" src/mcp/server.py; then echo "Unused: $v"; fi
done
```

## Red Flags

Stop and investigate if:

- Tests are skipped or commented out
- "TODO" or "FIXME" comments exist in completed code
- Documentation says "TBD" or "Coming soon"
- Acceptance criteria have caveats like "mostly done"
- Errors are caught but not handled properly
- Imports exist but functions aren't called

## Output Format

After verification, document what was checked:

```markdown
## Verification Report

### Task: [Task Name]

**Verified By**: [Agent/Human]
**Date**: [Date]

**Checks Performed**:
- [x] Implementation matches spec
- [x] Tests pass (X/Y tests)
- [x] Error handling verified
- [x] Documentation links valid

**Issues Found**: None / [List issues]

**Status**: Ready to mark complete / Needs work
```
