@echo off

set ROOT=%cd%
set SELF=%~dp0

rd /q /s "_NRD_SDK"
mkdir "_NRD_SDK"
cd "_NRD_SDK"

mkdir "Include"
mkdir "Integration"
mkdir "Lib\Debug"
mkdir "Lib\Release"
mkdir "Shaders"

copy "%SELF%\Include\*" "Include"
copy "%SELF%\Integration\*" "Integration"
copy "%SELF%\Shaders\Include\NRD.hlsli" "Shaders"
copy "%SELF%\Shaders\Include\NRDConfig.hlsli" "Shaders"
copy "%SELF%\LICENSE.txt" "."
copy "%SELF%\README.md" "."
copy "%SELF%\UPDATE.md" "."

copy "%ROOT%\_Bin\Debug\NRD.dll" "Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.lib" "Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.pdb" "Lib\Debug"
copy "%ROOT%\_Bin\Release\NRD.dll" "Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.lib" "Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.pdb" "Lib\Release"

cd ..

call "_Build\_deps\nri-src\3-PrepareSDK.bat"
