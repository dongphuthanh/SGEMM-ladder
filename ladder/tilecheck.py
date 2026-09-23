"""Sanity-check a cooperative-load mapping on the CPU, before writing the kernel.

Answers the two questions that matter, in about a second, with no GPU:
  1. coverage   - does every element of the tile get loaded exactly once?
  2. coalescing - how much of each fetched 32-byte sector does a warp actually use?

Add a check() call for any new mapping you're about to write.
"""

def check(name, H, W, NTHREADS, mapping, ld, warp=32, elem_bytes=4):
    """H x W  = tile shape in ELEMENTS (floats, or float4s if elem_bytes=16).
    ld = leading dimension of the source matrix, always in floats.
    mapping(t, off) -> (tileRow, tileCol) for thread t on pass `off`."""
    stride = NTHREADS // W
    passes = H // stride

    # --- coverage ---
    cover = {}
    for t in range(NTHREADS):
        for off in range(0, H, stride):
            rc = mapping(t, off)
            cover[rc] = cover.get(rc, 0) + 1
    missing = H * W - len({rc for rc in cover if 0 <= rc[0] < H and 0 <= rc[1] < W})
    dupes = sum(n - 1 for n in cover.values() if n > 1)
    oob = sum(1 for r, c in cover if not (0 <= r < H and 0 <= c < W))

    # --- coalescing: sectors one warp touches on a single load instruction ---
    addrs = [r * ld * 4 + c * elem_bytes for (r, c) in (mapping(t, 0) for t in range(warp))]
    sectors = len({a // 32 for a in addrs})
    eff = 100.0 * (warp * elem_bytes) / (sectors * 32)

    ok = not missing and not dupes and not oob
    status = "OK " if ok else f"BAD(miss {missing}, dup {dupes}, oob {oob})"
    print(f"{name:28s} stride={stride:3d} passes={passes:2d} | coverage {status} | "
          f"{sectors:2d} sectors/warp, {eff:5.1f}% used")


if __name__ == "__main__":
    BM, BN, BK, NTHREADS = 128, 128, 8, 256
    K, M = 4096, 4096

    # scalar loads (sharedthreadtilev2)
    check("As 128x8  scalar", BM, BK, NTHREADS,
          lambda t, off: (t // BK + off, t % BK), K)
    check("Bs 8x128  scalar", BK, BN, NTHREADS,
          lambda t, off: (t // BN + off, t % BN), M)

    # float4 loads (transpose / doublebuffer): tile measured in float4 units
    check("As 128x2  float4", BM, BK // 4, NTHREADS,
          lambda t, off: (t // (BK // 4) + off, t % (BK // 4)), K, elem_bytes=16)
    check("Bs 8x32   float4", BK, BN // 4, NTHREADS,
          lambda t, off: (t // (BN // 4) + off, t % (BN // 4)), M, elem_bytes=16)

    # BK = 16: two passes each, stride = NTHREADS / width
    check("As 128x4  float4 BK16", BM, 4, NTHREADS,
          lambda t, off: (t // 4 + off, t % 4), K, elem_bytes=16)
    check("Bs 16x32  float4 BK16", 16, BN // 4, NTHREADS,
          lambda t, off: (t // (BN // 4) + off, t % (BN // 4)), M, elem_bytes=16)
