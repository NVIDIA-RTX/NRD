@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

git -C "%ROOT%" submodule update --init --recursive
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

cmake -S "%ROOT%" -B "%ROOT%\_Build" -DNRD_NRI=ON %*
exit /B %ERRORLEVEL%
