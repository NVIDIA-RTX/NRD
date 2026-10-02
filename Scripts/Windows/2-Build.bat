@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

cmake --build "%ROOT%\_Build" --config Release -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

cmake --build "%ROOT%\_Build" --config Debug -j %NUMBER_OF_PROCESSORS%
exit /B %ERRORLEVEL%
