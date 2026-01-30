#!/bin/bash
# Setup llama.cpp for GGUF conversion
# This script clones and builds llama.cpp if not already present

set -e

echo "=================================================================="
echo "Setting up llama.cpp for GGUF conversion"
echo "=================================================================="
echo ""

# Check if llama.cpp already exists
if [ -d "llama.cpp" ]; then
    echo "✓ llama.cpp directory already exists"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd llama.cpp
        git pull
        echo "✓ Updated llama.cpp"
        cd ..
    fi
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
    echo "✓ Cloned llama.cpp"
fi

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo ""
    echo "✗ CMake not found"
    echo ""
    echo "Install CMake with:"
    echo "  macOS:   brew install cmake"
    echo "  Ubuntu:  sudo apt-get install cmake"
    echo "  Fedora:  sudo dnf install cmake"
    echo ""
    exit 1
fi

echo ""
echo "Building llama.cpp..."
cd llama.cpp

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build only the quantize tool (faster than building everything)
cmake --build . --target llama-quantize -j 8

cd ../..

echo ""
echo "=================================================================="
echo "✓ llama.cpp setup complete!"
echo "=================================================================="
echo ""
echo "Quantize binary location: llama.cpp/build/bin/llama-quantize"
echo ""
echo "You can now run: bash scripts/convert_mlx_to_gguf.sh"
echo ""
