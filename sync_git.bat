@echo off
setlocal
echo ========================================================
echo   Query My Meeting - Git Synchronization Tool
echo ========================================================
echo.

:: Ensure modal prompt is allowed for interactive credential popups
git config --local credential.modalPrompt true

echo [1/3] Checking Git status...
git status -s
echo.

echo [2/3] Staging changes...
git add .

set /p commit_msg="Enter commit message (or press ENTER for default): "
if "%commit_msg%"=="" set commit_msg="Update meeting transcript RAG project"

git commit -m "%commit_msg%"
echo.

echo [3/3] Pushing to GitHub (origin main)...
git push origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   SUCCESS: Changes successfully pushed to GitHub!
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   Push encountered an error. If authentication failed:
    echo   1. Ensure your GitHub Personal Access Token (PAT) is active.
    echo   2. Or switch to SSH remote:
    echo      git remote set-url origin git@github.com:amarjeetsingh92/Query-my-meeting.git
    echo ========================================================
)

echo.
pause
