@echo off

cd "_Build"

cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% NEQ 0 goto END

cmake --build . --config Debug -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% NEQ 0 goto END

:END
cd ..
exit /B %ERRORLEVEL%
