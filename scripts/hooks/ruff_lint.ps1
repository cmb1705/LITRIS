# Run ruff check and format on modified Python files.
# Called by PostToolUse hook when .py files are written or edited.
# Step 1: Auto-fix what ruff can fix (silently)
# Step 2: Auto-format (silently)
# Step 3: Check for remaining violations (output visible, exit non-zero if any)

param()

$path = $env:CLAUDE_FILE_PATH
if (-not $path) { exit 0 }
if ($path -notmatch '\.py$') { exit 0 }
if (-not (Test-Path $path)) { exit 0 }

# Auto-fix fixable violations (suppress output -- these are handled)
ruff check --fix $path 2>$null | Out-Null

# Auto-format
ruff format $path 2>$null | Out-Null

# Report remaining violations -- output goes to Claude, exit code signals pass/fail
ruff check $path
exit $LASTEXITCODE
