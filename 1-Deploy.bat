@echo off

git submodule update --init --recursive

mkdir "_Build"

cd "_Build"
cmake -DNRD_NRI=ON .. -A x64
cd ..
