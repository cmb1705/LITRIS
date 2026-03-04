# Lint markdown files on edit using markdownlint-cli2.
# Called by PostToolUse hook when .md files are edited.
# Usage: powershell -NoProfile -File scripts\hooks\lint_markdown.ps1

param()

$path = $env:CLAUDE_FILE_PATH
if (-not $path) { exit 0 }
if ($path -notmatch '\.md$') { exit 0 }

# Skip CLAUDE.md and other config-like markdown files that use non-standard formatting
if ($path -match 'CLAUDE\.md$|CLAUDE_SUPPLEMENTAL\.md$|ralph-loop\.local\.md$') { exit 0 }

# Run markdownlint-cli2 if available via npx
npx --yes markdownlint-cli2 $path 2>$null
exit 0
