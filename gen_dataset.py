"""
gen_dataset.py - generates a binary file of integers for the selection app.
 
File layout (little endian, explicit, so portable across machines):
  [int64  n]             number of elements that follow
  [int32  v0 .. vn-1]    the data
 
Usage: python3 gen_dataset.py <n> <output_file> [seed] [max_value] [--mode M]
  n          number of integers
  seed       RNG seed (default 42)
  max_value  values are in [0, max_value) (default 2^31-1, at most 2^31)
  --mode     random   (default) uniform random values
             sorted   increasing ramp, globally sorted across the file
             reverse  decreasing ramp
             The sorted/reverse modes make ranks lose their elements unevenly
             during selection, which is what exercises load balancing.
 
random mode uses splitmix64 and produces exactly the same file as the C, Lua,
Perl and Tcl generators for the same seed and max_value.
 
Uses numpy if it is installed (fast, ~50-100M ints/s); otherwise falls back to
pure standard-library Python (~1-2M ints/s).
"""
import argparse
import struct
import sys
from array import array

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB
CHUNK = 1 << 22

try:
    import numpy as np
except ImportError:
    np = None

def splitmix_np(seed, start, m, maxv):
    """Values start..start+m-1 of the splitmix64 stream, vectorized.
    splitmix64 is counter based: state_i = seed + (i+1)*GAMMA (mod 2^64)."""
    i = np.arange(start + 1, start + m + 1, dtype=np.uint64)
    z = np.uint64(seed & MASK) + i * np.uint64(GAMMA)  # wraps mod 2^64
    z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
    z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(maxv)).astype("<i4")
 
 
def splitmix_py(state, m, maxv):
    """Pure-Python version; returns (array of m int32, new state)."""
    out = array("i", bytes(4 * m))
    for j in range(m):
        state = (state + GAMMA) & MASK
        z = state
        z = ((z ^ (z >> 30)) * C1) & MASK
        z = ((z ^ (z >> 27)) * C2) & MASK
        out[j] = (z ^ (z >> 31)) % maxv
    return out, state
 
 
def ramp_np(start, m, n, maxv, reverse):
    i = np.arange(start, start + m, dtype=np.uint64)
    v = (i * np.uint64(maxv - 1)) // np.uint64(max(n - 1, 1))  # 0 .. maxv-1
    if reverse:
        v = np.uint64(maxv - 1) - v
    return v.astype("<i4")
 
 
def ramp_py(start, m, n, maxv, reverse):
    out = array("i", bytes(4 * m))
    d = max(n - 1, 1)
    for j in range(m):
        v = (start + j) * (maxv - 1) // d
        out[j] = (maxv - 1 - v) if reverse else v
    return out
 
 
def positive_int(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError("integers only")
    return v
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=positive_int)
    ap.add_argument("output_file")
    ap.add_argument("seed", type=positive_int, nargs="?", default=42)
    ap.add_argument("max_value", type=positive_int, nargs="?", default=2147483647)
    ap.add_argument("--mode", choices=["random", "sorted", "reverse"], default="random")
    a = ap.parse_args()
 
    if a.n <= 0 or a.max_value <= 0 or a.max_value > 2**31 or a.seed < 0:
        sys.exit("Invalid arguments (n > 0, 0 < max_value <= 2^31, seed >= 0)")
 
    big_endian = sys.byteorder == "big"
    state = a.seed & MASK
    done = 0
    try:
        f = open(a.output_file, "wb")
    except OSError as e:
        sys.exit(f"cannot open output file: {e}")
 
    with f:
        f.write(struct.pack("<q", a.n))
        while done < a.n:
            m = min(CHUNK, a.n - done)
            if a.mode == "random":
                if np is not None:
                    f.write(splitmix_np(a.seed, done, m, a.max_value).tobytes())
                else:
                    buf, state = splitmix_py(state, m, a.max_value)
                    if big_endian:
                        buf.byteswap()
                    f.write(buf.tobytes())
            else:
                rev = a.mode == "reverse"
                if np is not None:
                    f.write(ramp_np(done, m, a.n, a.max_value, rev).tobytes())
                else:
                    buf = ramp_py(done, m, a.n, a.max_value, rev)
                    if big_endian:
                        buf.byteswap()
                    f.write(buf.tobytes())
            done += m
 
    print(f"Wrote {a.n} integers to {a.output_file} ({(8 + 4 * a.n) / 2**20:.2f} MB)"
          f" [{a.mode}, {'numpy' if np is not None else 'pure python'}]")
 
 
if __name__ == "__main__":
    main()
