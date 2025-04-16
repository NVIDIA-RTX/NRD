@echo off

cd "_Build"
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
cmake --build . --config Debug -j %NUMBER_OF_PROCESSORS%
cd ..
