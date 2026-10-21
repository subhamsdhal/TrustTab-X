@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0Stop-TrustTabX.ps1"
set "rc=%errorlevel%"
if not "%rc%"=="0" (
  echo.
  echo TrustTab-X could not be stopped. Review the error above.
  pause
)
exit /b %rc%
