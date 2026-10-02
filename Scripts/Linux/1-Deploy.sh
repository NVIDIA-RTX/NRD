#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

git submodule update --init --recursive

cmake -S . -B _Build -DNRD_NRI=ON "$@"
