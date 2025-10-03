#!/bin/bash

ROOT=$(pwd)
SELF=$(dirname "$0")

rm -rf "_NRD_SDK"
mkdir -p "_NRD_SDK"
cd "_NRD_SDK"

mkdir -p "Include"
mkdir -p "Integration"
mkdir -p "Lib/Debug"
mkdir -p "Lib/Release"
mkdir -p "Shaders"

cp -r "$(SELF)/Include/" "Include"
cp -r "$(SELF)/Integration/" "Integration"
cp -r "$(SELF)/Shaders/Include/NRD.hlsli" "Shaders"
cp -r "$(SELF)/Shaders/Include/NRDConfig.hlsli" "Shaders"
cp "$(SELF)/LICENSE.txt" "."
cp "$(SELF)/README.md" "."
cp "$(SELF)/UPDATE.md" "."

cp -H "$(ROOT)/_Bin/Debug/libNRD.so" "Lib/Debug"
cp -H "$(ROOT)/_Bin/Release/libNRD.so" "Lib/Release"

cd ..

source "_Build\_deps\nri-src\3-PrepareSDK.sh"
