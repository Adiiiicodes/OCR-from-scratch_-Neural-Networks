#!/bin/bash
# setup_mnist.sh - Download and verify MNIST dataset for OCR-from-scratch project

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== MNIST Setup Script ===${NC}"

# --- 1. Create data directory ---
DATA_DIR="data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# --- 2. Download MNIST from S3 mirror (most reliable) ---
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo -e "${YELLOW}Downloading MNIST dataset...${NC}"
for f in "${FILES[@]}"; do
    echo -e "${YELLOW}→ Fetching $f${NC}"
    wget -q --show-progress -O "$f" "$BASE_URL/$f" || {
        echo -e "${RED}✗ Failed to download $f${NC}"
        exit 1
    }
done

# --- 3. Decompress ---
echo -e "${YELLOW}Decompressing...${NC}"
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        gunzip -f "$f"
        echo -e "${GREEN}✓ Extracted: ${f%.gz}${NC}"
    fi
done

cd ..

# --- 4. Verify with Python (embedded, not a bash command) ---
echo -e "${YELLOW}Verifying dataset integrity...${NC}"

python3 << 'PYEOF'
import struct
import os
import sys

def read_idx_header(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        dims = struct.unpack('>I', f.read(4))[0]
        shape = []
        for _ in range(dims):
            shape.append(struct.unpack('>I', f.read(4))[0])
    return magic, shape

checks = [
    ('data/train-images-idx3-ubyte', 0x803, [60000, 28, 28]),
    ('data/train-labels-idx1-ubyte', 0x801, [60000]),
    ('data/t10k-images-idx3-ubyte',  0x803, [10000, 28, 28]),
    ('data/t10k-labels-idx1-ubyte',  0x801, [10000]),
]

all_ok = True
for path, expected_magic, expected_shape in checks:
    if not os.path.exists(path):
        print(f"  ✗ MISSING: {path}")
        all_ok = False
        continue
    magic, shape = read_idx_header(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if magic == expected_magic and shape == expected_shape:
        print(f"  ✓ {os.path.basename(path):<30} magic={hex(magic)} shape={shape} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {os.path.basename(path):<30} got magic={hex(magic)} shape={shape}")
        all_ok = False

sys.exit(0 if all_ok else 1)
PYEOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== All MNIST files verified successfully ===${NC}"
    echo -e "${GREEN}Data location: $(pwd)/data/${NC}"
else
    echo -e "${RED}=== Verification failed ===${NC}"
    exit 1
fi
