#!/usr/bin/env python3
"""Warn when marking tasks as complete without verification."""
import os
import re
import sys

path = os.environ.get("CLAUDE_FILE_PATH", "")
old_string = os.environ.get("CLAUDE_OLD_STRING", "")
new_string = os.environ.get("CLAUDE_NEW_STRING", "")

is_task_file = bool(
    re.search(r"proposals[/\\]tasks", path)
    or re.search(r"ISSUES\.md", path)
    or re.search(r"STATE\.md", path)
)

marking_complete = "[ ]" in old_string and "[x]" in new_string

if is_task_file and marking_complete:
    YELLOW = "\033[33m"
    WHITE = "\033[37m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    RESET = "\033[0m"
    print()
    print(f"{YELLOW}============================================={RESET}")
    print(f"{YELLOW} TASK COMPLETION VERIFICATION REQUIRED{RESET}")
    print(f"{YELLOW}============================================={RESET}")
    print()
    print(f"{WHITE}You are marking a task/issue as complete.{RESET}")
    print(f"{WHITE}Before proceeding, verify:{RESET}")
    print()
    print(f"{CYAN}  [1] Implementation matches specification{RESET}")
    print(f"{CYAN}  [2] All acceptance criteria are met{RESET}")
    print(f"{CYAN}  [3] Tests pass and cover the functionality{RESET}")
    print(f"{CYAN}  [4] Error handling returns structured responses{RESET}")
    print(f"{CYAN}  [5] All imported functions/validators are used{RESET}")
    print(f"{CYAN}  [6] Documentation links resolve correctly{RESET}")
    print(f"{CYAN}  [7] No dead code or unused imports{RESET}")
    print()
    print(f"{RED}If any criteria are not met, fix them first!{RESET}")
    print(f"{YELLOW}============================================={RESET}")
    print()

sys.exit(0)
