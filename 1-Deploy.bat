@echo off

git submodule update --init --recursive

mkdir "_Build"

cd "_Build"

cmake -DNRD_NRI=ON .. %*
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

cd ..
