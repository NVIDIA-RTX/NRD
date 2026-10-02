@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

set "SDK=%ROOT%\_NRD_SDK"
set "NRI=%ROOT%\_Build\_deps\nri-src"
set "NRI_SDK=%ROOT%\_NRI_SDK"

echo %SDK%: ROOT=%ROOT%

if exist "%SDK%" rd /q /s "%SDK%"

mkdir "%SDK%\Include"
mkdir "%SDK%\Integration"
mkdir "%SDK%\Lib\Debug"
mkdir "%SDK%\Lib\Release"
mkdir "%SDK%\Shaders"

copy "%ROOT%\Include\*" "%SDK%\Include"
copy "%ROOT%\Integration\*" "%SDK%\Integration"
copy "%ROOT%\Shaders\NRD.hlsli" "%SDK%\Shaders"
copy "%ROOT%\Shaders\NRDConfig.hlsli" "%SDK%\Shaders"
copy "%ROOT%\LICENSE.txt" "%SDK%"
copy "%ROOT%\README.md" "%SDK%"
copy "%ROOT%\UPDATE.md" "%SDK%"

copy "%ROOT%\_Bin\Debug\NRD.dll" "%SDK%\Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.lib" "%SDK%\Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.pdb" "%SDK%\Lib\Debug"
copy "%ROOT%\_Bin\Release\NRD.dll" "%SDK%\Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.lib" "%SDK%\Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.pdb" "%SDK%\Lib\Release"

if exist "%NRI%\Include\NRI.h" (
    if exist "%NRI_SDK%" rd /q /s "%NRI_SDK%"
    mkdir "%NRI_SDK%\Include\Extensions"
    mkdir "%NRI_SDK%\Lib\Debug"
    mkdir "%NRI_SDK%\Lib\Release"

    copy "%NRI%\Include\*" "%NRI_SDK%\Include"
    copy "%NRI%\Include\Extensions\*" "%NRI_SDK%\Include\Extensions"
    copy "%NRI%\LICENSE.txt" "%NRI_SDK%"
    copy "%NRI%\README.md" "%NRI_SDK%"
    copy "%NRI%\nri.natvis" "%NRI_SDK%"

    copy "%ROOT%\_Bin\Debug\NRI.dll" "%NRI_SDK%\Lib\Debug"
    copy "%ROOT%\_Bin\Debug\NRI.lib" "%NRI_SDK%\Lib\Debug"
    copy "%ROOT%\_Bin\Debug\NRI.pdb" "%NRI_SDK%\Lib\Debug"
    copy "%ROOT%\_Bin\Release\NRI.dll" "%NRI_SDK%\Lib\Release"
    copy "%ROOT%\_Bin\Release\NRI.lib" "%NRI_SDK%\Lib\Release"
)
