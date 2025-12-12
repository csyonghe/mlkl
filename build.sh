#!/bin/bash
# Build script for Slang Machine Learning Kernel Library

set -e  # Exit on any error

BUILD_DIR="build"
CMAKE_GENERATOR="Unix Makefiles"

echo "Building Slang Machine Learning Kernel Library..."

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

# Configure the project
echo "Configuring project..."
cmake -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

echo "Build completed successfully!"
echo "Executables can be found in: $BUILD_DIR/"
