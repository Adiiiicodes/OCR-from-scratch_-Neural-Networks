import struct
import os

def read_idx_header(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n_dims = struct.unpack('>I', f.read(4))[0]
        shape = [struct.unpack('>I', f.read(4))[0] for _ in range(n_dims)]
    return magic, shape

checks = [
    ('data/train-images-idx3-ubyte', 0x803, [60000, 28, 28]),
    ('data/train-labels-idx1-ubyte', 0x801, [60000]),
    ('data/t10k-images-idx3-ubyte',  0x803, [10000, 28, 28]),
    ('data/t10k-labels-idx1-ubyte',  0x801, [10000]),
]

all_ok = True
for path, exp_magic, exp_shape in checks:
    if not os.path.exists(path):
        print(f"  X MISSING: {path}")
        all_ok = False
        continue
    magic, shape = read_idx_header(path)
    ok = (magic == exp_magic and shape == exp_shape)
    mark = 'OK' if ok else 'X '
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {mark} {path}: magic={hex(magic)} shape={shape} ({size_mb:.1f} MB)")
    if not ok:
        all_ok = False

print("\n" + ("ALL OK" if all_ok else "SOME FILES BAD"))