# External Dependencies - Best Practices

This document explains why certain dependencies are NOT included in the repository and how to set them up.

---

## llama.cpp - GGUF Conversion Tool

### Status: ❌ NOT Included in Repository

**Location**: Must be cloned separately to project root

**Why NOT included?**

1. **Size** (~500 MB)
   - Contains compiled binaries and build artifacts
   - Would bloat the repository unnecessarily
   - Slows down git operations

2. **External Project**
   - Has its own version control (https://github.com/ggerganov/llama.cpp)
   - Actively maintained with frequent updates
   - Users should get the latest version directly

3. **Platform-Specific**
   - Compiled binaries are OS/architecture-specific
   - macOS binaries won't work on Linux/Windows
   - Users need to build for their platform anyway

4. **Best Practice**
   - External dependencies should be installed, not bundled
   - Similar to how we don't check in `node_modules/` or `venv/`
   - Keeps repository clean and focused

---

## How to Setup

### Automated Setup (Recommended)

```bash
bash scripts/setup_llama_cpp.sh
```

This script will:
- Clone llama.cpp if not present
- Check if CMake is installed
- Build only the quantize tool (faster)
- Verify the build succeeded

### Manual Setup

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp

# Build with CMake
cd llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build quantize tool (or build all with: cmake --build . -j 8)
cmake --build . --target llama-quantize -j 8

cd ../..
```

### Prerequisites

- **CMake**: Required for building
  - macOS: `brew install cmake`
  - Ubuntu: `sudo apt-get install cmake`
  - Fedora: `sudo dnf install cmake`

---

## When Is It Needed?

### Phase 1: Data Collection
❌ **NOT needed**

### Phase 2: Training Data Generation
❌ **NOT needed**

### Phase 3: Model Training
❌ **NOT needed** (MLX training only)

### Phase 3: Model Export (GGUF Conversion)
✅ **REQUIRED** - For converting trained models to GGUF format for Ollama deployment

---

## .gitignore Configuration

The repository is configured to exclude `llama.cpp/`:

```gitignore
# External dependencies (clone separately)
llama.cpp/
```

This prevents accidentally committing:
- The entire llama.cpp repository
- Compiled binaries
- Build artifacts
- Platform-specific files

---

## Alternative: Python-Only Approach

For users who don't want to build llama.cpp, there's a Python-only alternative:

### Using llama.cpp Python Package

```bash
pip install llama-cpp-python
```

However, this still requires compilation and may not support all features. The recommended approach is to use the official llama.cpp repository.

---

## Troubleshooting

### "cmake not found"

**Solution**: Install CMake
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake

# Fedora/RHEL
sudo dnf install cmake
```

### "Build failed"

**Common causes**:
- Missing C++ compiler
- Outdated CMake version
- Insufficient permissions

**Solution**: Install build tools
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"
```

### "llama.cpp already exists"

**Solution**: Update it
```bash
cd llama.cpp
git pull
cd build
cmake --build . --target llama-quantize -j 8
cd ../..
```

---

## Best Practices for External Dependencies

### DO Include in Repository:
✅ Configuration files (e.g., `requirements.txt`)
✅ Setup scripts (e.g., `setup_llama_cpp.sh`)
✅ Documentation on how to install
✅ Version requirements/pinning

### DO NOT Include in Repository:
❌ The actual external dependency
❌ Compiled binaries
❌ Build artifacts
❌ Platform-specific files
❌ Large third-party repositories

### Document Instead:
- Where to get the dependency
- How to install it
- What version is required
- Platform-specific instructions
- Troubleshooting common issues

---

## Similar Examples in Other Projects

This approach follows industry standards:

- **Node.js projects**: Don't commit `node_modules/`
- **Python projects**: Don't commit `venv/` or `.venv/`
- **Ruby projects**: Don't commit `vendor/`
- **Go projects**: Don't commit `vendor/` (pre-modules)
- **Rust projects**: Don't commit `target/`

**Key principle**: Dependencies should be declared and installed, not bundled.

---

## Summary

✅ **llama.cpp is excluded** from the repository
✅ **.gitignore is configured** to prevent accidental commits
✅ **Setup script is provided** for easy installation
✅ **Documentation is updated** with setup instructions
✅ **Follows best practices** for dependency management

This keeps the repository:
- **Small** and fast to clone
- **Clean** and focused on project code
- **Portable** across platforms
- **Up-to-date** with latest llama.cpp version

---

**Last Updated**: January 30, 2026
**See Also**:
- `docs/GGUF_EXPORT_GUIDE.md` - GGUF conversion documentation
- `scripts/setup_llama_cpp.sh` - Automated setup script
- `scripts/convert_mlx_to_gguf.sh` - GGUF conversion script
