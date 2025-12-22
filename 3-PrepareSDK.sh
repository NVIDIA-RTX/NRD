#!/bin/bash

ROOT=$(pwd)
SELF=$(dirname "$0")
SDK=_NRD_SDK

echo ${SDK}: ROOT=${ROOT}, SELF=${SELF}

rm -rf "${SDK}"

mkdir -p "${SDK}/Include"
mkdir -p "${SDK}/Integration"
mkdir -p "${SDK}/Lib/Debug"
mkdir -p "${SDK}/Lib/Release"
mkdir -p "${SDK}/Shaders"

cp -r "${SELF}/Include/." "${SDK}/Include"
cp -r "${SELF}/Integration/." "${SDK}/Integration"
cp "${SELF}/Shaders/Include/NRD.hlsli" "${SDK}/Shaders"
cp "${SELF}/Shaders/Include/NRDConfig.hlsli" "${SDK}/Shaders"
cp "${SELF}/LICENSE.txt" "${SDK}/"
cp "${SELF}/README.md" "${SDK}/"
cp "${SELF}/UPDATE.md" "${SDK}/"

# Find the _Bin directory regardless of where we are
BIN_DIR=$(find $(pwd) -type d -name "_Bin" | head -n 1)

if [ -z "$BIN_DIR" ]; then
    echo "ERROR: _Bin directory not found anywhere!"
    # List files to see what happened
    ls -R
    exit 1
fi

echo "Found _Bin at: $BIN_DIR"
