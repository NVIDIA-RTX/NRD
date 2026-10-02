@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

if exist "%ROOT%\build" rd /q /s "%ROOT%\build"
if exist "%ROOT%\_Bin" rd /q /s "%ROOT%\_Bin"
if exist "%ROOT%\_Build" rd /q /s "%ROOT%\_Build"
if exist "%ROOT%\_Shaders" rd /q /s "%ROOT%\_Shaders"
if exist "%ROOT%\_NRD_SDK" rd /q /s "%ROOT%\_NRD_SDK"
if exist "%ROOT%\_NRI_SDK" rd /q /s "%ROOT%\_NRI_SDK"
