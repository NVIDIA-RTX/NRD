#!/bin/bash

ROOT=$(pwd)
SELF=$(cd "$(dirname "$0")" && pwd)

rm -rf "_NRD_SDK"

mkdir -p "_NRD_SDK/Include"
mkdir -p "_NRD_SDK/Integration"
mkdir -p "_NRD_SDK/Lib/Debug"
mkdir -p "_NRD_SDK/Lib/Release"
mkdir -p "_NRD_SDK/Shaders"

cp -r "${SELF}/Include/." "_NRD_SDK/Include"
cp -r "${SELF}/Integration/." "_NRD_SDK/Integration"
cp "${SELF}/Shaders/Include/NRD.hlsli" "_NRD_SDK/Shaders"
cp "${SELF}/Shaders/Include/NRDConfig.hlsli" "_NRD_SDK/Shaders"
cp "${SELF}/LICENSE.txt" "_NRD_SDK/"
cp "${SELF}/README.md" "_NRD_SDK/"
cp "${SELF}/UPDATE.md" "_NRD_SDK/"

cp -H "${ROOT}/_Bin/Debug/libNRD.so" "_NRD_SDK/Lib/Debug"
cp -H "${ROOT}/_Bin/Release/libNRD.so" "_NRD_SDK/Lib/Release"

if [ -f "_Build/_deps/nri-src/3-PrepareSDK.sh" ]; then
    bash "_Build/_deps/nri-src/3-PrepareSDK.sh"
fi
