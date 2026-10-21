@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0Start-TrustTabX.ps1"
set "rc=%errorlevel%"
if not "%rc%"=="0" (
  echo.
  echo TrustTab-X did not start. Review the error above.
  pause
)
exit /b %rc%
