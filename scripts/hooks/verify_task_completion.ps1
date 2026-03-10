# Task Completion Verification Hook
# This script warns when marking tasks as complete without verification

$path = $env:CLAUDE_FILE_PATH
$oldString = $env:CLAUDE_OLD_STRING
$newString = $env:CLAUDE_NEW_STRING

# Check if this is a task/issue file being edited
$isTaskFile = $path -match 'proposals[/\\]tasks' -or
              $path -match 'ISSUES\.md' -or
              $path -match 'STATE\.md'

# Check if we're marking something as complete
$markingComplete = $oldString -match '\[ \]' -and $newString -match '\[x\]'

if ($isTaskFile -and $markingComplete) {
    Write-Host ""
    Write-Host "=============================================" -ForegroundColor Yellow
    Write-Host " TASK COMPLETION VERIFICATION REQUIRED" -ForegroundColor Yellow
    Write-Host "=============================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You are marking a task/issue as complete." -ForegroundColor White
    Write-Host "Before proceeding, verify:" -ForegroundColor White
    Write-Host ""
    Write-Host "  [1] Implementation matches specification" -ForegroundColor Cyan
    Write-Host "  [2] All acceptance criteria are met" -ForegroundColor Cyan
    Write-Host "  [3] Tests pass and cover the functionality" -ForegroundColor Cyan
    Write-Host "  [4] Error handling returns structured responses" -ForegroundColor Cyan
    Write-Host "  [5] All imported functions/validators are used" -ForegroundColor Cyan
    Write-Host "  [6] Documentation links resolve correctly" -ForegroundColor Cyan
    Write-Host "  [7] No dead code or unused imports" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "If any criteria are not met, fix them first!" -ForegroundColor Red
    Write-Host "=============================================" -ForegroundColor Yellow
    Write-Host ""
}

# Always exit 0 - this is a warning, not a blocker
exit 0
