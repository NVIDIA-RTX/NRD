#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

cmake -S . -B _Build
cmake --build _Build --config Release --parallel
cmake --build _Build --config Debug --parallel
