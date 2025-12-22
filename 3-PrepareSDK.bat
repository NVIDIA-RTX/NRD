@echo off

set ROOT=%cd%
set SELF=%~dp0
set SDK=_NRD_SDK

echo %SDK%: ROOT=%ROOT%, SELF=%SELF%

rd /q /s "%SDK%"

mkdir "%SDK%\Include"
mkdir "%SDK%\Integration"
mkdir "%SDK%\Lib\Debug"
mkdir "%SDK%\Lib\Release"
mkdir "%SDK%\Shaders"

copy "%SELF%\Include\*" "%SDK%\Include"
copy "%SELF%\Integration\*" "%SDK%\Integration"
copy "%SELF%\Shaders\Include\NRD.hlsli" "%SDK%\Shaders"
copy "%SELF%\Shaders\Include\NRDConfig.hlsli" "%SDK%\Shaders"
copy "%SELF%\LICENSE.txt" "%SDK%"
copy "%SELF%\README.md" "%SDK%"
copy "%SELF%\UPDATE.md" "%SDK%"

copy "%ROOT%\_Bin\Debug\NRD.dll" "%SDK%\Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.lib" "%SDK%\Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.pdb" "%SDK%\Lib\Debug"
copy "%ROOT%\_Bin\Release\NRD.dll" "%SDK%\Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.lib" "%SDK%\Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.pdb" "%SDK%\Lib\Release"
