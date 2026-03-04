# Run pytest for test files related to a modified src/ module.
# Called by PostToolUse hook when src/**/*.py files are edited.
# Usage: powershell -NoProfile -File scripts\hooks\run_related_tests.ps1

param()

$path = $env:CLAUDE_FILE_PATH
if (-not $path) { exit 0 }
if ($path -notmatch 'src[/\\].*\.py$') { exit 0 }
if ($path -match '__pycache__') { exit 0 }
if ($path -match '__init__\.py$') { exit 0 }

# Extract the top-level module name from src/<module>/... or src/<module>.py
$relative = $path -replace '.*src[/\\]', ''
$parts = $relative -split '[/\\]'
$module = $parts[0] -replace '\.py$', ''

# Also get the file stem for direct match
$stem = [System.IO.Path]::GetFileNameWithoutExtension($path)

# Build candidate test file names (most specific first)
$candidates = @()
if ($stem -ne $module) {
    $candidates += "tests/test_$stem.py"
}
$candidates += "tests/test_$module.py"

foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        pytest $candidate -v --tb=short -x 2>$null
        exit $LASTEXITCODE
    }
}

# No matching test file found -- that's fine, just exit silently
exit 0
