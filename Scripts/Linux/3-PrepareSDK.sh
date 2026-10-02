#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

SDK=_NRD_SDK
NRI=_Build/_deps/nri-src
NRI_SDK=_NRI_SDK

echo "$SDK: ROOT=$PWD"

rm -rf "$SDK"
mkdir -p "$SDK/Include" "$SDK/Integration" "$SDK/Lib" "$SDK/Shaders"

cp -r Include/. "$SDK/Include"
cp -r Integration/. "$SDK/Integration"
cp Shaders/NRD.hlsli Shaders/NRDConfig.hlsli "$SDK/Shaders"
cp LICENSE.txt README.md UPDATE.md "$SDK"
cp -H _Bin/libNRD.so "$SDK/Lib"

if [ -f "$NRI/Include/NRI.h" ]; then
    rm -rf "$NRI_SDK"
    mkdir -p "$NRI_SDK/Include" "$NRI_SDK/Lib"

    cp -r "$NRI/Include/." "$NRI_SDK/Include"
    cp "$NRI/LICENSE.txt" "$NRI/README.md" "$NRI/nri.natvis" "$NRI_SDK"
    cp -H _Bin/libNRI.so "$NRI_SDK/Lib"
fi
