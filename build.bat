@echo off
REM Build script for Slang Machine Learning Kernel Library

set BUILD_DIR=build
set CMAKE_GENERATOR="Visual Studio 17 2022"

REM Create build directory if it doesn't exist
if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

REM Configure the project
echo Configuring project...
cmake -B %BUILD_DIR% -G %CMAKE_GENERATOR% -A x64

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b %ERRORLEVEL%
)

REM Build the project
echo Building project...
cmake --build %BUILD_DIR% --config Release --parallel

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo Build completed successfully!
echo Executables can be found in: %BUILD_DIR%\Release\
