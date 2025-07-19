@echo off

cd "_Build"

cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

cmake --build . --config Debug -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

cd ..
