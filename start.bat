@echo off
REM ============================================================
REM  ProofLens / DEEP DIVE - start the local web server and
REM  open the landing screen in your default browser.
REM
REM  Keep the "ProofLens Server" window open while browsing.
REM  Closing that window stops the server.
REM ============================================================
setlocal
cd /d "%~dp0"

echo Starting ProofLens server on http://127.0.0.1:8000/ ...
start "ProofLens Server" cmd /k python -m app.main

REM Give the server a moment to boot, then open the browser.
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8000/

echo.
echo Browser opened. If it says it cannot connect, wait a moment
echo and refresh. Stop the server by closing the server window.
endlocal