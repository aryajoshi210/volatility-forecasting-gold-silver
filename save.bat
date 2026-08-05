@echo off
REM Saves your work: stages all changes, commits with your message, and pushes to GitHub.
REM Usage: save.bat "what you changed"

if "%~1"=="" (
    echo Usage: save.bat "description of what you changed"
    exit /b 1
)

git add -A
git commit -m "%~1"
git push
