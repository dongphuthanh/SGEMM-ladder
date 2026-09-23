"""Generate the SGEMM tiling figure page as static HTML + inline SVG."""

# ------------------------------------------------------------ svg helpers
def R(x, y, w, h, cls="", rx=0, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}" {extra}/>'

def T(x, y, s, cls="m", anchor="start", size=11, extra=""):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" '
            f'font-size="{size}" {extra}>{s}</text>')

def L(x1, y1, x2, y2, cls="ln", extra=""):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" {extra}/>'

def AR(x1, y1, x2, y2, fig, cls="ln", extra=""):
    return L(x1, y1, x2, y2, cls, f'marker-end="url(#ar{fig})" {extra}')

def PATH(d, fig, cls="ln", arrow=True):
    m = f'marker-end="url(#ar{fig})"' if arrow else ""
    return f'<path d="{d}" class="{cls}" fill="none" {m}/>'

def DOTS(x, y, vertical=True, n=3, gap=5):
    out = []
    for i in range(n):
        cx, cy = (x, y + i * gap) if vertical else (x + i * gap, y)
        out.append(f'<circle cx="{cx}" cy="{cy}" r="1.3" class="dot"/>')
    return "".join(out)

def svg(fig, w, h, body, label):
    defs = (f'<defs><marker id="ar{fig}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
            f'xmlns="http://www.w3.org/2000/svg">{defs}{body}</svg>')


# ======================================================================
# FIG 1 : one block, one slab
# ======================================================================
def fig1():
    b = []
    S = 200            # matrix square
    BX, BY = 360, 40   # B
    AX, AY = 90, 280   # A
    CX, CY = 360, 280  # C
    band = 50          # 128 of 512
    by, bx = 1, 2
    slab = 10          # drawn width of an 8-wide slab (not to scale)
    si = 60            # slab drawn offset

    # matrices
    for (x, y, name, dims, cls) in [(BX, BY, "B", "K x N", "fb3"), (AX, AY, "A", "M x K", "fa3"),
                                    (CX, CY, "C", "M x N", "fc3")]:
        b.append(R(x, y, S, S, cls + " box"))

    # bands and slabs
    b.append(R(AX, AY + by * band, S, band, "fa2"))                 # A row band
    b.append(R(AX + si, AY + by * band, slab, band, "fa1"))         # A slab
    b.append(R(BX + bx * band, BY, band, S, "fb2"))                 # B col band
    b.append(R(BX + bx * band, BY + si, band, slab, "fb1"))         # B slab
    b.append(R(CX + bx * band, CY + by * band, band, band, "fc1"))  # C tile

    # block grid lines, drawn on top so they read over the fills
    for i in range(1, 4):
        for (x, y) in [(BX, BY), (AX, AY), (CX, CY)]:
            b.append(L(x + i * band, y, x + i * band, y + S, "blk"))
            b.append(L(x, y + i * band, x + S, y + i * band, "blk"))
    for (x, y, name, dims) in [(BX, BY, "B", "K x N"), (AX, AY, "A", "M x K"), (CX, CY, "C", "M x N")]:
        b.append(T(x + 8, y + 18, name, "m bold", size=15))
        b.append(T(x + S - 8, y + 18, dims, "m mut", "end", 11))

    # dimension labels
    b.append(T(AX - 10, AY + S / 2 + 4, "M = 512", "m mut", "end", 11))
    b.append(T(AX + S / 2, AY + S + 16, "K = 512", "m mut", "middle", 11))
    b.append(T(BX - 10, BY + S / 2 + 4, "K = 512", "m mut", "end", 11))
    b.append(T(BX + S / 2, BY - 10, "N = 512", "m mut", "middle", 11))
    b.append(T(CX + S / 2, CY + S + 16, "N = 512", "m mut", "middle", 11))

    # annotations
    b.append(T(AX + 4, AY + by * band + 14, "row band", "m", size=9))
    b.append(T(AX + 4, AY + by * band + 26, "BM = 128", "m", size=9))
    b.append(T(BX + bx * band + band / 2, BY + S + 14, "col band, BN = 128", "m", "middle", 11))
    b.append(T(CX + bx * band + band / 2, CY + by * band + band / 2 - 2, "128", "m on1", "middle", 10))
    b.append(T(CX + bx * band + band / 2, CY + by * band + band / 2 + 10, "x 128", "m on1", "middle", 10))
    b.append(T(CX + bx * band + band / 2, CY + by * band + band + 14, "this block's tile", "m", "middle", 9))
    b.append(T(CX + bx * band + band / 2, CY + by * band + band + 26, "(bx=2, by=1)", "m mut", "middle", 9))

    # slab pointers
    b.append(AR(AX + si + slab / 2, AY + S + 30, AX + si + slab / 2, AY + by * band + band + 3, 1))
    b.append(T(AX + si + slab / 2, AY + S + 44, "slab i: BK = 8 cols", "m", "middle", 10))
    b.append(AR(BX + bx * band - 30, BY + si + slab / 2, BX + bx * band - 3, BY + si + slab / 2, 1))
    b.append(T(BX + bx * band - 34, BY + si + slab / 2 + 4, "slab i: BK = 8 rows", "m", "end", 10))

    # shared tiles on the right
    TX, TW, TH = 680, 256, 32
    ASY, BSY = 300, 100
    for (y, name, c2, c1, axis) in [(BSY, "Bs[8][128]", "fb2", "fb1", "k down, C-col across"),
                                    (ASY, "As[8][128] transposed", "fa2", "fa1", "k down, C-row across")]:
        b.append(R(TX, y, TW, TH, c2 + " box"))
        b.append(R(TX, y, 16, TH, c1))
        b.append(T(TX, y - 8, name, "m bold", size=12))
        b.append(T(TX - 6, y + TH / 2 + 4, "8", "m mut", "end", 10))
        b.append(T(TX + TW / 2, y + TH + 14, "128   (" + axis + ")", "m mut", "middle", 10))
    b.append(T(TX + TW / 2, ASY + TH + 28, "same 8 x 128 shape as Bs", "m", "middle", 10))

    # B slab -> Bs (straight right)
    b.append(PATH(f"M{BX + bx * band + band},{BY + si + slab / 2} C 600,{BY + si + slab / 2} 600,{BSY + TH / 2} {TX - 4},{BSY + TH / 2}", 1))
    b.append(T(600, BSY - 30, "1 float4 load / thread", "m", "middle", 10))
    b.append(T(600, BSY - 18, "straight store", "m mut", "middle", 10))

    # A slab -> As, routed UNDER C so it never crosses it
    ym = AY + by * band + band / 2
    b.append(PATH(f"M{AX + si + slab},{ym} L318,{ym} Q328,{ym} 328,{ym + 10} L328,505 Q328,515 338,515 "
                  f"L650,515 Q660,515 660,505 L660,{ASY + TH / 2 + 10} Q660,{ASY + TH / 2} 670,{ASY + TH / 2} L{TX - 4},{ASY + TH / 2}", 1))
    b.append(T(494, 506, "1 float4 load / thread   -   scatter-store = transpose", "m", "middle", 10))

    # K loop note
    b.append(T(AX, 548, "for i in 0 .. K/BK-1:  slab slides along K in A, down K in B", "m mut", size=10))

    return svg(1, 960, 565, "".join(b),
               "One thread block owns a 128 by 128 tile of C. For each 8-wide slab along K it loads a 128 by 8 piece of A and an 8 by 128 piece of B into shared memory.")


# ======================================================================
# FIG 2 : loading As with float4 + transpose
# ======================================================================
def fig2():
    b = []
    c = 14                       # cell
    # ---- left: A slab [128][8], rows 0-3 and 126-127 drawn
    LX, LY = 70, 70
    rows = [0, 1, 2, 3, None, 126, 127]
    ry = {}
    y = LY
    for r in rows:
        if r is None:
            b.append(DOTS(LX + 4 * c, y + 4, True))
            y += 22
            continue
        ry[r] = y
        y += c
    for k in range(8):
        b.append(T(LX + k * c + c / 2, LY - 8, f"k{k}", "m mut", "middle", 9))
    b.append(T(LX + 4 * c, LY - 24, "slab of A  [128 rows][8 k]", "m bold", "middle", 12))

    tint = {0: "fa1", 1: "fa2", 2: "fb1"}   # thread -> fill
    ontxt = {0: "on1", 1: "", 2: "on1"}
    for r, yy in ry.items():
        b.append(T(LX - 8, yy + c - 4, f"row {r}", "m mut", "end", 9))
        for half in range(2):
            t = 2 * r + half
            cls = tint.get(t, "fa3")
            b.append(R(LX + half * 4 * c, yy, 4 * c, c, cls + " f4"))
            b.append(T(LX + half * 4 * c + 2 * c, yy + c - 4, f"t{t}", "m " + ontxt.get(t, ""), "middle", 9))
        # cell ticks
        for k in range(1, 8):
            if k != 4:
                b.append(L(LX + k * c, yy, LX + k * c, yy + c, "tick"))
    b.append(T(LX + 4 * c, ry[127] + c + 16, "2 float4 per row  ->  2 threads per row", "m", "middle", 10))
    b.append(T(LX + 4 * c, ry[127] + c + 30, "innerRowA = t / 2      innerColA = t % 2", "m mut", "middle", 10))

    # ---- right: As [8][128], cols 0-3 and 126-127 drawn
    RX, RY = 520, 70
    cols = [0, 1, 2, 3, None, 126, 127]
    cx = {}
    x = RX
    for cc in cols:
        if cc is None:
            b.append(DOTS(x + 4, RY + 4 * c, False))
            x += 22
            continue
        cx[cc] = x
        x += c
    for k in range(8):
        b.append(T(RX - 8, RY + k * c + c - 4, f"k{k}", "m mut", "end", 9))
    b.append(T(RX + 3.5 * c + 11, RY - 24, "As  [8 k][128 cols]", "m bold", "middle", 12))
    for cc, xx in cx.items():
        b.append(T(xx + c / 2, RY - 8, f"{cc}", "m mut", "middle", 9))
        for k in range(8):
            # which thread wrote this cell: column cc = innerRowA -> t = 2*cc + (k//4)
            t = 2 * cc + (k // 4)
            cls = tint.get(t, "fa3")
            b.append(R(xx, RY + k * c, c, c, cls + " cell"))
    # thread labels on the right grid (vertical groups)
    b.append(T(cx[0] + c / 2, RY + 2 * c + 4, "t0", "m on1", "middle", 9))
    b.append(T(cx[0] + c / 2, RY + 6 * c + 4, "t1", "m", "middle", 9))
    b.append(T(cx[1] + c / 2, RY + 2 * c + 4, "t2", "m on1", "middle", 9))
    b.append(T(RX + 3.5 * c + 11, RY + 8 * c + 16, "column = innerRowA  (which row of A)", "m", "middle", 10))
    b.append(T(RX + 3.5 * c + 11, RY + 8 * c + 30, "rows k = innerColA*4 .. +3", "m mut", "middle", 10))

    # ---- arrows: t0, t1, t2
    b.append(PATH(f"M{LX + 4 * c},{ry[0] + c / 2} C 330,{ry[0] + c / 2} 400,{RY + 2 * c} {cx[0] - 3},{RY + 2 * c}", 2))
    b.append(PATH(f"M{LX + 8 * c},{ry[0] + c / 2} C 360,{ry[0] + c / 2} 420,{RY + 6 * c} {cx[0] - 3},{RY + 6 * c}", 2))
    b.append(PATH(f"M{LX + 4 * c},{ry[1] + c / 2} C 330,{ry[1] + c / 2} 420,{RY + 2 * c + 6} {cx[1] - 3},{RY + 2 * c + 6}", 2, "ln2"))
    b.append(T(330, 42, "tmp.x tmp.y tmp.z tmp.w", "m", "middle", 10))
    b.append(T(330, 55, "4 k-values from ONE row of A", "m mut", "middle", 9))
    b.append(T(440, 175, "land in 4 ROWS,", "m", "middle", 10))
    b.append(T(440, 188, "one column of As", "m mut", "middle", 9))

    return svg(2, 660, 300, "".join(b),
               "Each thread loads one float4, four consecutive k values from one row of A, and stores them down one column of the transposed As tile.")


# ======================================================================
# FIG 3 : loading Bs with float4, no transpose
# ======================================================================
def fig3():
    b = []
    sw, sh = 26, 16              # slot (4 floats) drawn size
    LX, LY = 70, 60
    slots = [0, 1, 2, 3, None, 30, 31]
    sx = {}
    x = LX
    for s in slots:
        if s is None:
            b.append(DOTS(x + 4, LY + 4 * sh, False))
            x += 22
            continue
        sx[s] = x
        x += sw
    gridW = x - LX
    b.append(T(LX + gridW / 2, LY - 24, "slab of B  [8 k][128 cols]  =  Bs, same layout", "m bold", "middle", 12))
    for s, xx in sx.items():
        b.append(T(xx + sw / 2, LY - 8, f"col {s * 4}", "m mut", "middle", 8))
    for k in range(8):
        b.append(T(LX - 8, LY + k * sh + sh - 5, f"k{k}", "m mut", "end", 9))
        for s, xx in sx.items():
            t = 32 * k + s
            cls = "fb1" if t == 0 else ("fb2" if k == 0 else "fb3")
            on = "on1" if t == 0 else ""
            b.append(R(xx, LY + k * sh, sw, sh, cls + " f4"))
            b.append(T(xx + sw / 2, LY + k * sh + sh - 5, f"t{t}", "m " + on, "middle", 8))
    b.append(T(LX + gridW / 2, LY + 8 * sh + 18, "32 float4 per row  ->  32 threads per row, 8 rows  =  256 threads, one pass", "m", "middle", 10))
    b.append(T(LX + gridW / 2, LY + 8 * sh + 32, "innerRowB = t / 32      innerColB = (t % 32) * 4", "m mut", "middle", 10))

    # right note
    NX = 560
    b.append(T(NX, LY + 10, "no transpose needed", "m bold", size=12))
    b.append(T(NX, LY + 28, "B's contiguous axis is already", "m", size=10))
    b.append(T(NX, LY + 41, "the C-column axis, so the tile", "m", size=10))
    b.append(T(NX, LY + 54, "is k-major as loaded.", "m", size=10))
    b.append(T(NX, LY + 78, "one 128-bit load, one 128-bit", "m mut", size=10))
    b.append(T(NX, LY + 91, "store, thread t0 -> Bs[0][0..3]", "m mut", size=10))
    b.append(T(NX, LY + 115, "a warp (t0..t31) fills a whole", "m mut", size=10))
    b.append(T(NX, LY + 128, "row: 512 contiguous bytes of B", "m mut", size=10))

    return svg(3, 920, 260, "".join(b),
               "Each thread loads one float4 from B and stores it in place; Bs needs no transpose because B is already contiguous along the C-column axis.")


# ======================================================================
# FIG 4 : compute - one thread's 8x8 outer product
# ======================================================================
def fig4():
    b = []
    # strips
    SX, W128 = 70, 256          # 2 px per column
    px = W128 / 128
    rh = 6
    dot = 3
    tR, tC = 5, 9               # threadRow, threadCol  -> t = 89
    a0, b0 = tR * 8, tC * 8     # 40, 72

    def strip(y, name, cls, hlcls, hl0):
        b.append(R(SX, y, W128, 8 * rh, cls + " box"))
        for k in range(1, 8):
            b.append(L(SX, y + k * rh, SX + W128, y + k * rh, "tick"))
        b.append(R(SX, y + dot * rh, W128, rh, "hlrow"))
        b.append(R(SX + hl0 * px, y + dot * rh, 8 * px, rh, hlcls))
        b.append(T(SX - 8, y + 8 * rh / 2 + 4, name, "m bold", "end", 11))
        b.append(T(SX - 8, y + dot * rh + 5, f"k={dot}", "m mut", "end", 8))
        b.append(T(SX + W128 + 6, y + dot * rh + 5, f"row dot={dot}", "m mut", size=9))

    ASY, BSY = 40, 100
    strip(ASY, "As", "fa3", "fa1", a0)
    b.append(T(SX + a0 * px + 8, ASY - 6, f"cols {a0}..{a0+7} = threadRow*8..+7", "m", "middle", 9))
    strip(BSY, "Bs", "fb3", "fb1", b0)
    b.append(T(SX + b0 * px + 8, BSY + 8 * rh + 14, f"cols {b0}..{b0+7} = threadCol*8..+7", "m", "middle", 9))

    # C tile 16x16 grid
    CX, CY, cc = 70, 200, 14
    b.append(T(CX, CY - 10, "C tile 128x128 = 16x16 threads, 8x8 each", "m bold", size=11))
    b.append(R(CX + tC * cc, CY, cc, 16 * cc, "fc2"))               # column highlight
    b.append(R(CX, CY + tR * cc, 16 * cc, cc, "fc2"))               # row highlight
    for i in range(17):
        b.append(L(CX + i * cc, CY, CX + i * cc, CY + 16 * cc, "tick2"))
        b.append(L(CX, CY + i * cc, CX + 16 * cc, CY + i * cc, "tick2"))
    b.append(R(CX, CY, 16 * cc, 16 * cc, "box"))
    b.append(R(CX + tC * cc, CY + tR * cc, cc, cc, "fc1"))
    b.append(T(CX + tC * cc + cc / 2, CY + tR * cc + cc - 4, "89", "m on1", "middle", 8))
    b.append(T(CX - 6, CY + tR * cc + cc - 4, f"threadRow {tR}", "m mut", "end", 8))
    b.append(T(CX + tC * cc + cc / 2, CY + 16 * cc + 12, f"threadCol {tC}", "m mut", "middle", 8))
    b.append(T(CX + 16 * cc + 8, CY + tR * cc + 6, "threads 80..95", "m", size=9))
    b.append(T(CX + 16 * cc + 8, CY + tR * cc + 18, "all read the SAME", "m mut", size=8))
    b.append(T(CX + 16 * cc + 8, CY + tR * cc + 28, "As[3][40..47]", "m mut", size=8))
    b.append(T(CX + 16 * cc + 8, CY + tR * cc + 40, "(broadcast, free)", "m mut", size=8))
    b.append(T(CX + tC * cc + cc / 2, CY + 16 * cc + 24, "t = 89 = 5*16 + 9", "m", "middle", 9))
    b.append(T(CX + tC * cc + cc / 2, CY + 16 * cc + 36, "threadRow = t/16, threadCol = t%16", "m mut", "middle", 8))

    # outer product
    OX, OY, oc = 560, 230, 16
    b.append(T(OX + 4 * oc + 20, OY - 40, "one dot iteration: rank-1 outer product", "m bold", "middle", 11))
    # b row vector above
    for j in range(8):
        b.append(R(OX + oc + j * oc, OY - oc - 4, oc, oc, "fb1 cell"))
    b.append(T(OX + oc + 4 * oc, OY - oc - 8, "b[0..7]  =  Bs[3][72..79]", "m", "middle", 9))
    # a column vector left
    for i in range(8):
        b.append(R(OX - 4, OY + i * oc, oc, oc, "fa1 cell"))
    b.append(T(OX - 26, OY + 4 * oc + 4, "a[0..7] = As[3][40..47]", "m", "middle", 9, 'transform="rotate(-90 ' + str(OX - 26) + ' ' + str(OY + 4 * oc + 4) + ')"'))
    # acc grid
    for i in range(8):
        for j in range(8):
            b.append(R(OX + oc + j * oc, OY + i * oc, oc, oc, "fc3 cell"))
    b.append(T(OX + oc + 4 * oc, OY + 4 * oc + 4, "acc[i][j] += a[i] * b[j]", "m", "middle", 10))
    b.append(T(OX + oc + 4 * oc, OY + 8 * oc + 16, "8 + 8 shared reads  ->  64 FMAs", "m", "middle", 10))
    b.append(T(OX + oc + 4 * oc, OY + 8 * oc + 30, "x 8 dot values per slab, x K/8 slabs", "m mut", "middle", 9))

    # arrows strips -> vectors, acc -> C cell
    b.append(PATH(f"M{SX + a0 * px + 8 * px},{ASY + dot * rh + rh / 2} C 420,{ASY + dot * rh} 500,{OY + 2 * oc} {OX - 6},{OY + 2 * oc}", 4))
    b.append(PATH(f"M{SX + b0 * px + 8 * px},{BSY + dot * rh + rh / 2} C 460,{BSY + dot * rh} {OX + oc + 4 * oc},{OY - 60} {OX + oc + 4 * oc},{OY - oc - 20}", 4))
    b.append(PATH(f"M{OX + oc},{OY + 8 * oc + 40} C 480,{OY + 8 * oc + 60} 300,{CY + tR * cc + cc + 20} {CX + tC * cc + cc + 2},{CY + tR * cc + cc - 2}", 4))
    b.append(T(CX + 8 * cc, CY + 16 * cc + 52, "after the K loop: C[blockRow+40+i][blockCol+72+j] = acc[i][j]", "m mut", "middle", 9))

    return svg(4, 920, 500, "".join(b),
               "Thread 89 owns the 8 by 8 block at thread row 5, thread column 9. Per dot step it reads 8 values from row dot of As and 8 from row dot of Bs and accumulates their outer product.")



# ======================================================================
# FIG 5 : timeline - single buffer vs prefetch vs double buffer
# ======================================================================
def fig5():
    b = []
    LDG, STS, BAR, CMP, PRO = 60, 14, 8, 100, 82     # px per phase
    X0 = 190                                          # timeline origin
    lanes = [("single buffer", "as written", 50), ("register prefetch", "one buffer", 200), ("double buffer", "two buffers", 350)]
    LH = 108
    TOP, MID, BOT = 6, 40, 78                          # row offsets inside a lane

    def bar(x, y, w, h, cls, label="", lcls="m on1", size=9):
        b.append(R(x, y, w, h, cls, rx=2))
        if label:
            b.append(T(x + w / 2, y + h / 2 + 3, label, lcls, "middle", size))

    def barrier(x, y):
        b.append(R(x, y + 2, BAR, LH - 4, "barw"))
        b.append(T(x + BAR / 2, y + LH + 10, "BAR", "m mut", "middle", 7))

    def prologue(x, y):
        bar(x, y + TOP, LDG, 24, "fa2 ph", "LDG slab 0", "m", 8)
        bar(x + LDG, y + BOT, STS, 22, "fb1 ph", "", "m on1", 8)
        barrier(x + LDG + STS, y)
        b.append(T(x + PRO / 2, y - 6, "prologue", "m mut", "middle", 8))
        return x + PRO

    for (name, sub, y) in lanes:
        b.append(T(20, y + 46, name, "m bold", size=12))
        b.append(T(20, y + 60, sub, "m mut", size=10))
        b.append(L(X0, y + LH + 2, 900, y + LH + 2, "tick"))
        b.append(T(X0 - 8, y + TOP + 15, "LDG", "m mut", "end", 8))
        b.append(T(X0 - 8, y + MID + 17, "FFMA", "m mut", "end", 8))
        b.append(T(X0 - 8, y + BOT + 14, "STS", "m mut", "end", 8))

    # ---- lane 1: single buffer ----
    y = lanes[0][2]; x = X0
    for i in range(3):
        bar(x, y + TOP, LDG, 24, "fa2 ph", f"LDG slab {i}", "m", 8)
        b.append(T(x + LDG / 2, y + MID + 17, "idle", "m mut", "middle", 8))
        bar(x + LDG, y + BOT, STS, 22, "fb1 ph")
        barrier(x + LDG + STS, y)
        bar(x + LDG + STS + BAR, y + MID, CMP, 28, "fc1 ph", f"compute {i}", "m on1", 9)
        barrier(x + LDG + STS + BAR + CMP, y)
        x += LDG + STS + BAR + CMP + BAR
    end1 = x
    b.append(T(X0 + LDG / 2, y + TOP - 2, "no FMAs while the load is in flight", "m", "start", 8))

    # ---- lane 2: register prefetch, one buffer ----
    y = lanes[1][2]; x = prologue(X0, y)
    for i in range(3):
        if i < 2:
            bar(x, y + TOP, LDG, 24, "fa2 ph", f"LDG slab {i+1}", "m", 8)
        bar(x, y + MID, CMP, 28, "fc1 ph", f"compute {i}", "m on1", 9)
        barrier(x + CMP, y)
        if i < 2:
            bar(x + CMP + BAR, y + BOT, STS, 22, "fb1 ph")
        barrier(x + CMP + BAR + STS, y)
        x += CMP + BAR + STS + BAR
    end2 = x
    b.append(T(X0 + PRO + 2 * (CMP + BAR + STS + BAR) + 4, y + TOP + 15, "<- each LDG in flight under a compute", "m", "start", 8))

    # ---- lane 3: double buffer ----
    y = lanes[2][2]; x = prologue(X0, y)
    for i in range(3):
        if i < 2:
            bar(x, y + TOP, LDG, 24, "fa2 ph", f"LDG slab {i+1}", "m", 8)
        bar(x, y + MID, CMP, 28, "fc1 ph", f"compute {i}  buf {i%2}", "m on1", 9)
        if i < 2:
            bar(x + CMP, y + BOT, STS, 22, "fb1 ph")
            b.append(T(x + CMP + STS + 4, y + BOT + 14, f"-> buf {(i+1)%2}", "m mut", "start", 7))
        barrier(x + CMP + STS, y)
        x += CMP + STS + BAR
    end3 = x

    # ---- total-time brackets ----
    for (end, y, txt) in [(end1, lanes[0][2], "3 slabs"),
                          (end2, lanes[1][2], f"{100 - int(100*(end2-X0)/(end1-X0))}% shorter"),
                          (end3, lanes[2][2], f"{100 - int(100*(end3-X0)/(end1-X0))}% shorter, one BAR per slab")]:
        b.append(L(X0, y + LH + 14, end, y + LH + 14, "ln2"))
        b.append(L(end, y + LH + 10, end, y + LH + 18, "ln"))
        b.append(T(end + 6, y + LH + 17, txt, "m", "start", 9))

    b.append(T(X0, 24, "time  ->", "m mut", "start", 10))
    b.append(T(900, 24, "bar lengths are illustrative, not measured", "m mut", "end", 9))
    return svg(5, 940, 500, "".join(b),
               "Timeline of three K-loop schemes over three slabs. Single buffer serialises load, store, barrier, compute. Register prefetch overlaps the global load with compute but keeps two barriers. Double buffering keeps the overlap and drops to one barrier per slab.")


# ======================================================================
# FIG 6 : buffer ownership per slab - why one barrier is enough
# ======================================================================
def fig6():
    b = []
    GX, GY = 250, 90
    CW, CH = 150, 58
    PW = 100
    b.append(T(GX - PW, GY - 58, "which buffer each slab READS (compute) and WRITES (store slab i+1)", "m bold", size=12))
    b.append(T(GX - PW - 8, GY + CH / 2 + 4, "buf 0", "m bold", "end", 11))
    b.append(T(GX - PW - 8, GY + CH + CH / 2 + 4, "buf 1", "m bold", "end", 11))
    # prologue column
    b.append(R(GX - PW, GY, PW, CH, "cellb fb2"))
    b.append(T(GX - PW / 2, GY + CH / 2 - 3, "WRITE", "m", "middle", 9))
    b.append(T(GX - PW / 2, GY + CH / 2 + 10, "slab 0", "m mut", "middle", 8))
    b.append(R(GX - PW, GY + CH, PW, CH, "cellb"))
    b.append(T(GX - PW / 2, GY + CH + CH / 2 + 4, "-", "m mut", "middle", 9))
    b.append(T(GX - PW / 2, GY - 8, "prologue", "m mut", "middle", 9))
    # slab columns
    for i in range(4):
        x = GX + i * CW
        b.append(T(x + CW / 2, GY - 8, f"slab {i}", "m mut", "middle", 9))
        rd = i % 2
        for row in range(2):
            y = GY + row * CH
            if row == rd:
                b.append(R(x, y, CW, CH, "cellb fa2"))
                b.append(T(x + CW / 2, y + CH / 2 - 3, "READ", "m", "middle", 9))
                b.append(T(x + CW / 2, y + CH / 2 + 10, f"compute {i}", "m mut", "middle", 8))
            else:
                if i < 3:
                    b.append(R(x, y, CW, CH, "cellb fb2"))
                    b.append(T(x + CW / 2, y + CH / 2 - 3, "WRITE", "m", "middle", 9))
                    b.append(T(x + CW / 2, y + CH / 2 + 10, f"slab {i+1}", "m mut", "middle", 8))
                else:
                    b.append(R(x, y, CW, CH, "cellb"))
                    b.append(T(x + CW / 2, y + CH / 2 + 4, "-", "m mut", "middle", 9))
    # barrier lines between columns
    for i in range(4):
        x = GX + i * CW
        b.append(L(x, GY - 14, x, GY + 2 * CH + 14, "ln2"))
        b.append(T(x, GY + 2 * CH + 26, "BAR", "m mut", "middle", 8))

    # hazard arrows across the first barrier
    b.append(PATH(f"M{GX + CW * 0.75},{GY + 6} C {GX + CW * 0.9},{GY - 30} {GX + CW * 1.1},{GY - 30} {GX + CW * 1.25},{GY + 6}", 6))
    b.append(T(GX + CW, GY - 34, "overwrite hazard", "m", "middle", 9))
    b.append(PATH(f"M{GX + CW * 0.75},{GY + 2 * CH - 6} C {GX + CW * 0.9},{GY + 2 * CH + 46} {GX + CW * 1.1},{GY + 2 * CH + 46} {GX + CW * 1.25},{GY + 2 * CH - 6}", 6))
    b.append(T(GX + CW, GY + 2 * CH + 60, "visibility hazard", "m", "middle", 9))

    EX = GX - PW
    b.append(T(EX, GY + 2 * CH + 84, "Both arrows cross the same barrier. It guarantees:", "m", size=10))
    b.append(T(EX, GY + 2 * CH + 100, "  every read of buf 0 in slab 0 finished   ->  slab 1 may overwrite it", "m mut", size=10))
    b.append(T(EX, GY + 2 * CH + 114, "  every write to buf 1 in slab 0 is visible  ->  slab 1 may read it", "m mut", size=10))
    b.append(T(EX, GY + 2 * CH + 134, "A single buffer needs a second barrier between READ and WRITE of the SAME buffer.", "m", size=10))
    return svg(6, 940, 360, "".join(b),
               "Buffer ownership across slabs: each slab reads one buffer and writes the other, alternating. The one barrier between slabs covers both the overwrite hazard and the visibility hazard.")



# ======================================================================
# FIG 7 : warptiling - warp footprint on the C tile, and what it pulls from shared
# ======================================================================
def fig7():
    b = []
    cc = 15                      # px per 8x8 thread tile  -> 16 cells = 240 px
    G = 16 * cc                  # tile side in px
    PX = [70, 540]               # panel origins (x of the As strip)
    PY = 118                     # y of tile top
    SW = 18                      # As strip width
    BH = 18                      # Bs strip height

    def panel(px, title, sub, warp_of, lane_labels, as_rows, bs_cols, counts):
        tx = px + SW + 10        # tile x
        ty = PY
        b.append(T(tx + G / 2, ty - 60, title, "m bold", "middle", 12))
        b.append(T(tx + G / 2, ty - 46, sub, "m mut", "middle", 9))
        # Bs strip above the tile (its 128 axis = C columns)
        bsy = ty - BH - 10
        b.append(R(tx, bsy, G, BH, "fb3 box"))
        b.append(R(tx, bsy, G * bs_cols / 128, BH, "fb1"))
        b.append(T(tx - 6, bsy + BH / 2 + 4, "Bs[dot]", "m bold", "end", 9))
        b.append(T(tx + G + 6, bsy + BH / 2 + 4, f"{bs_cols} distinct", "m", "start", 9))
        # As strip left of the tile (its 128 axis = C rows)
        b.append(R(px, ty, SW, G, "fa3 box"))
        b.append(R(px, ty, SW, G * as_rows / 128, "fa1"))
        b.append(T(px + SW / 2, ty + G + 12, "As[dot]", "m bold", "middle", 9))
        b.append(T(px + SW / 2, ty + G + 24, f"{as_rows}", "m", "middle", 9))
        b.append(T(px + SW / 2, ty + G + 35, "distinct", "m", "middle", 8))
        # C tile: colour each 8x8 thread tile by the warp that owns it
        for r in range(16):
            for c in range(16):
                w = warp_of(r, c)
                cls = "fc1" if w == 0 else ("fc2" if w % 2 else "fc3")
                b.append(R(tx + c * cc, ty + r * cc, cc, cc, cls + " cell"))
        b.append(R(tx, ty, G, G, "box"))
        for (r, c, lab) in lane_labels:
            b.append(T(tx + c * cc + cc / 2, ty + r * cc + cc - 4, lab, "m on1", "middle", 7))
        b.append(T(tx + G / 2, ty + G + 14, "C tile: 16 x 16 thread tiles of 8 x 8", "m mut", "middle", 8))
        b.append(T(tx + G / 2, ty + G + 26, "warp 0 = solid, other warps alternate", "m mut", "middle", 8))
        # counts
        for i, line in enumerate(counts):
            b.append(T(tx + G / 2, ty + G + 48 + i * 13, line, "m bold" if i == 2 else "m", "middle", 9))

    # left: current mapping  threadRow = t/16, threadCol = t%16  -> warp = t/32 = threadRow/2
    panel(PX[0], "now:  threadRow = t / 16,  threadCol = t % 16",
          "warp w owns thread-rows 2w, 2w+1  ->  16 rows x 128 cols of C",
          lambda r, c: r // 2,
          [(0, 0, "t0"), (0, 15, "t15"), (1, 0, "t16"), (1, 15, "t31")],
          16, 128,
          ["per dot step, warp 0 touches", "16 As + 128 Bs = 144 floats", "2048 FMAs / 144 = 14.2 per float"])

    # right: warptiled  warp 2x4 over tile, lanes 8x4 inside warp
    panel(PX[1], "warptiled:  warp 2 x 4,  lanes 8 x 4",
          "warp w owns a 64 x 32 block  ->  same 2048 outputs, squarer",
          lambda r, c: (r // 8) * 4 + (c // 4),
          [(0, 0, "t0"), (0, 3, "t3"), (1, 0, "t4"), (7, 3, "t31")],
          64, 32,
          ["per dot step, warp 0 touches", "64 As + 32 Bs = 96 floats", "2048 FMAs / 96 = 21.3 per float"])

    # centre note
    b.append(T(470, PY + 40, "same per-thread", "m", "middle", 9))
    b.append(T(470, PY + 53, "8 + 8 loads,", "m mut", "middle", 9))
    b.append(T(470, PY + 66, "64 FMAs", "m mut", "middle", 9))
    b.append(T(470, PY + 90, "only WHO owns", "m", "middle", 9))
    b.append(T(470, PY + 103, "which tile", "m", "middle", 9))
    b.append(T(470, PY + 116, "changes", "m", "middle", 9))
    b.append(AR(440, PY + 140, 500, PY + 140, 7))
    b.append(T(470, PY + 160, "rows + cols", "m mut", "middle", 8))
    b.append(T(470, PY + 172, "144 -> 96", "m bold", "middle", 10))
    return svg(7, 940, 445, "".join(b),
               "Warptiling changes which thread owns which 8 by 8 tile so that one warp covers a 64 by 32 block of C instead of a 16 by 128 strip. Per dot step the warp then needs 96 distinct floats from shared memory instead of 144, for the same 2048 FMAs.")


# ======================================================================
# FIG 8 : bank phases for the Bs fragment read (LDS.128), lanes 0-7
# ======================================================================
def fig8():
    b = []
    BX0, BW = 150, 22            # bank boxes
    def banks(y, title, lanes, verdict, vcls):
        b.append(T(20, y + 4, title, "m bold", size=11))
        b.append(T(20, y + 18, "phase 0 = lanes 0..7,", "m mut", size=8))
        b.append(T(20, y + 29, "b[0..3] = Bs[dot][threadCol*8 + 0..3]", "m mut", size=8))
        for k in range(32):
            b.append(R(BX0 + k * BW, y + 46, BW, 18, "cellb"))
            if k % 4 == 0:
                b.append(T(BX0 + k * BW + 2, y + 59, f"{k}", "m mut", "start", 7))
        b.append(T(BX0 - 6, y + 59, "bank", "m mut", "end", 8))
        # lane bars: row A = lanes 0-3, row B = lanes 4-7
        for (lane, tcol, row, cls, note) in lanes:
            bank0 = (tcol * 8) % 32
            yy = y + 72 + row * 22
            b.append(R(BX0 + bank0 * BW, yy, 4 * BW, 18, cls + " ph", rx=2))
            b.append(T(BX0 + bank0 * BW + 2 * BW, yy + 12, f"t{lane}  {note}", "m on1" if cls != "hlrow" else "m", "middle", 8))
        b.append(T(BX0 - 6, y + 72 + 12, "lanes 0-3", "m mut", "end", 8))
        b.append(T(BX0 - 6, y + 94 + 12, "lanes 4-7", "m mut", "end", 8))
        b.append(T(BX0 + 16 * BW, y + 130, verdict, vcls, "middle", 10))

    # now: lanes 0-7 -> threadCol 0-7 -> chunks at bytes tcol*32, banks (tcol*8)%32
    lanes_now = []
    for lane in range(8):
        tcol = lane
        row = 0 if lane < 4 else 1
        cls = "fb1" if row == 0 else "hlrow"
        lanes_now.append((lane, tcol, row, cls, f"@{tcol * 32}B"))
    banks(40, "now", lanes_now,
          "lanes 0 and 4 hit banks 0-3 at DIFFERENT addresses (0 B vs 128 B)  ->  2-way conflict, phase runs twice",
          "m bold")

    # warptiled: lanes 0-7 -> laneRow 0,0,0,0,1,1,1,1  laneCol 0,1,2,3,0,1,2,3 -> threadCol = laneCol
    lanes_wt = []
    for lane in range(8):
        tcol = lane % 4
        row = 0 if lane < 4 else 1
        cls = "fb1" if row == 0 else "fb2"
        lanes_wt.append((lane, tcol, row, cls, f"@{tcol * 32}B"))
    banks(210, "warptiled", lanes_wt,
          "lanes 0 and 4 hit banks 0-3 at the SAME address (0 B)  ->  broadcast, phase runs once",
          "m bold")
    return svg(8, 940, 370, "".join(b),
               "Shared memory serves a 128-bit warp load in phases of 8 lanes. Today lanes 0 and 4 map to different addresses in the same four banks, a 2-way conflict. Under warptiling they map to the same address, which the hardware broadcasts for free.")


# ======================================================================
# PAGE
# ======================================================================
CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  --bg:#F4F5F7; --panel:#FFFFFF; --ink:#1B1F26; --mut:#5E6772; --line:#C8CDD5; --grid:#DDE1E7;
  --a1:#0E8A8C; --a2:#8FCFD0; --a3:#D9EFEF;
  --b1:#C4581C; --b2:#EDB08A; --b3:#F8E2D3;
  --c1:#4A56C9; --c2:#C3C8F0; --c3:#E6E8F9;
  --hl:#F6D77A;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#12151A; --panel:#1A1E25; --ink:#E7E9EE; --mut:#98A1AE; --line:#39404B; --grid:#2B313B;
  --a1:#3FC4C6; --a2:#1D6C6E; --a3:#12393A;
  --b1:#F08D4E; --b2:#7E4520; --b3:#3B2213;
  --c1:#8F99F2; --c2:#3A428A; --c3:#242A52;
  --hl:#8A6F1E;
}}
:root[data-theme="dark"]{
  --bg:#12151A; --panel:#1A1E25; --ink:#E7E9EE; --mut:#98A1AE; --line:#39404B; --grid:#2B313B;
  --a1:#3FC4C6; --a2:#1D6C6E; --a3:#12393A;
  --b1:#F08D4E; --b2:#7E4520; --b3:#3B2213;
  --c1:#8F99F2; --c2:#3A428A; --c3:#242A52;
  --hl:#8A6F1E;
}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55;
     padding-block:32px 64px;padding-inline:20px;}
main{max-width:940px;margin:0 auto;display:grid;gap:40px;}
header h1{font-size:28px;font-weight:600;margin:0 0 6px;letter-spacing:-0.01em;text-wrap:balance;}
header p{margin:0;color:var(--mut);max-width:65ch;}
.params{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:14px;font-family:var(--mono);font-size:12.5px;}
.params span b{font-weight:600;color:var(--ink);} .params span{color:var(--mut);}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;color:var(--mut);margin-top:10px;}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:6px;}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:20px 20px 16px;}
figure > div{overflow-x:auto;}
figure svg{display:block;width:100%;max-width:100%;height:auto;min-width:640px;color:var(--ink);font-family:var(--mono);}
figcaption{margin-top:12px;font-size:14px;color:var(--ink);max-width:70ch;}
figcaption .n{font-family:var(--mono);font-size:11.5px;color:var(--mut);letter-spacing:.06em;text-transform:uppercase;display:block;margin-bottom:4px;}
pre{margin:14px 0 0;background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:12px 14px;
    font-family:var(--mono);font-size:12.5px;line-height:1.5;overflow-x:auto;color:var(--ink);}
pre .c{color:var(--mut);}
.m{fill:var(--ink)} .mut{fill:var(--mut)} .bold{font-weight:600}
.on1{fill:var(--bg)}
.ln{stroke:currentColor;stroke-width:1.2} .ln2{stroke:currentColor;stroke-width:1.2;stroke-dasharray:3 3}
.grid{stroke:var(--grid);stroke-width:1} .tick{stroke:var(--line);stroke-width:.8}
.box{stroke:var(--ink);stroke-width:1;fill:none}
.blk{stroke:var(--mut);stroke-width:.9;opacity:.65}
.tick2{stroke:var(--line);stroke-width:.9}
.f4{stroke:var(--ink);stroke-width:1} .cell{stroke:var(--panel);stroke-width:.6}
.dot{fill:var(--mut)}
.fa1{fill:var(--a1)} .fa2{fill:var(--a2)} .fa3{fill:var(--a3)}
.fb1{fill:var(--b1)} .fb2{fill:var(--b2)} .fb3{fill:var(--b3)}
.fc1{fill:var(--c1)} .fc2{fill:var(--c2)} .fc3{fill:var(--c3)}
.hlrow{fill:var(--hl);fill-opacity:.45}
.ph{stroke:var(--panel);stroke-width:1}
.barw{fill:var(--hl);fill-opacity:.55}
.cellb{stroke:var(--line);stroke-width:1;fill:none}
.cellb.fa2{fill:var(--a2)} .cellb.fb2{fill:var(--b2)}
@media (max-width:600px){ figure{padding:14px 12px;} body{font-size:14px;} }
</style>
"""

HTML = f"""<title>SGEMM Tile Map</title>
{CSS}
<main>
<header>
  <h1>SGEMM Tile Map</h1>
  <p>How one thread block turns a 128&times;128 tile of C into 256 threads&rsquo; worth of 8&times;8 register tiles, with the float4 loads and the transpose on the way into shared memory. Drawn for the config in <code>sharedthreadtilev2</code>; matrices shown at a demo size of 512 so the four blocks per side are visible.</p>
  <div class="params">
    <span><b>BM</b> = 128</span><span><b>BN</b> = 128</span><span><b>BK</b> = 8</span>
    <span><b>TM</b> = 8</span><span><b>TN</b> = 8</span><span><b>threads</b> = (BM/TM)&times;(BN/TN) = 256</span>
    <span><b>C</b> = A&middot;B, A is M&times;K, B is K&times;N</span>
  </div>
  <div class="legend">
    <span><i style="background:var(--a1)"></i>A / As</span>
    <span><i style="background:var(--b1)"></i>B / Bs</span>
    <span><i style="background:var(--c1)"></i>C / acc</span>
    <span><i style="background:var(--hl)"></i>the row <code>dot</code> being read</span>
  </div>
</header>

<figure>
  <div>{fig1()}</div>
  <figcaption><span class="n">1 &middot; block level</span>
  Block (bx=2, by=1) owns one 128&times;128 tile of C. It never sees all of A or B &mdash; only its row band of A and column band of B, and only an 8-wide slab of each at a time. Both slabs land in shared memory as 8&times;128 tiles, the same shape, because As is transposed on the way in.</figcaption>
<pre>blockRow = blockIdx.y * BM;   blockCol = blockIdx.x * BN;
for (i = 0; i &lt; K/BK; i++) {{          <span class="c">// slab i: A cols i*8..+7, B rows i*8..+7</span>
    load slab of A -&gt; As   (fig 2)
    load slab of B -&gt; Bs   (fig 3)
    __syncthreads();
    compute            (fig 4)
    __syncthreads();
}}</pre>
</figure>

<figure>
  <div>{fig2()}</div>
  <figcaption><span class="n">2 &middot; loading As</span>
  A row of the A slab is 8 floats = 2 float4s, so 2 threads cover a row and 256 threads cover all 128 rows in one pass. Thread t reads 4 consecutive k-values from row t/2 and writes them down <em>column</em> t/2 of As, rows (t%2)&middot;4..+3. That scatter is the transpose &mdash; A itself is never touched.</figcaption>
<pre>innerRowA = t / 2;   innerColA = t % 2;
aRow = blockRow + innerRowA;   aCol = i*BK + innerColA*4;
float4 tmp = *reinterpret_cast&lt;const float4*&gt;(&amp;A[aRow*K + aCol]);
As[innerColA*4 + 0][innerRowA] = tmp.x;    <span class="c">// k = innerColA*4</span>
As[innerColA*4 + 1][innerRowA] = tmp.y;
As[innerColA*4 + 2][innerRowA] = tmp.z;
As[innerColA*4 + 3][innerRowA] = tmp.w;</pre>
</figure>

<figure>
  <div>{fig3()}</div>
  <figcaption><span class="n">3 &middot; loading Bs</span>
  A row of the B slab is 128 floats = 32 float4s, so 32 threads cover a row and 8 rows take exactly 256 threads. Same coverage rule as As &mdash; divide by the tile width in float4 units &mdash; but the divisor is 32 instead of 2 because the tile is wide instead of tall. No transpose: the float4 goes in as one store.</figcaption>
<pre>innerRowB = t / 32;   innerColB = (t % 32) * 4;
bRow = i*BK + innerRowB;   bCol = blockCol + innerColB;
float4 tmp = *reinterpret_cast&lt;const float4*&gt;(&amp;B[bRow*N + bCol]);
*reinterpret_cast&lt;float4*&gt;(&amp;Bs[innerRowB][innerColB]) = tmp;</pre>
</figure>

<figure>
  <div>{fig4()}</div>
  <figcaption><span class="n">4 &middot; compute</span>
  Thread 89 sits at threadRow 5, threadCol 9 and owns C rows 40..47 &times; cols 72..79 of the tile. For each of the 8 <code>dot</code> values in a slab it pulls 8 contiguous floats from row <code>dot</code> of As and 8 from row <code>dot</code> of Bs, then does 64 FMAs. The 16 threads sharing threadRow 5 all read the same As strip; shared memory broadcasts it. The 16 sharing threadCol 9 all read the same Bs strip.</figcaption>
<pre>threadRow = t / 16;   threadCol = t % 16;             <span class="c">// t = 89 -&gt; 5, 9</span>
for (dot = 0; dot &lt; BK; dot++) {{
    a[j] = As[dot][threadRow*8 + j];     j = 0..7    <span class="c">// contiguous -&gt; 2 x LDS.128</span>
    b[k] = Bs[dot][threadCol*8 + k];     k = 0..7    <span class="c">// contiguous -&gt; 2 x LDS.128</span>
    acc[j][k] += a[j] * b[k];            8 x 8 = 64 FMAs
}}</pre>
</figure>

<figure>
  <div>{fig5()}</div>
  <figcaption><span class="n">5 &middot; pipelining the K loop</span>
  Three ways to run the slab loop. As written, nothing computes while the global load is in flight and every slab pays two barriers. Register prefetch issues the next slab&rsquo;s load before computing the current one, so the latency hides under 512 FMAs &mdash; but with one shared buffer you still need a barrier before you can overwrite it, then another after. Double buffering writes the next slab into the <em>other</em> buffer, so the first of those barriers disappears.</figcaption>
<pre>load slab 0 -&gt; buf[0];  __syncthreads();                     <span class="c">// prologue</span>
for (i = 0; i &lt; numTiles; i++) {{
    cur = i % 2;  nxt = (i + 1) % 2;
    if (i + 1 &lt; numTiles)  tmpA, tmpB = LDG slab i+1;        <span class="c">// issue, don't wait</span>
    compute slab i from As[cur], Bs[cur];                    <span class="c">// hides the LDG</span>
    if (i + 1 &lt; numTiles)  STS tmpA, tmpB -&gt; As[nxt], Bs[nxt];
    __syncthreads();                                          <span class="c">// the only barrier</span>
}}</pre>
</figure>

<figure>
  <div>{fig6()}</div>
  <figcaption><span class="n">6 &middot; why one barrier is enough</span>
  Slab i reads buffer i&nbsp;%&nbsp;2 and writes buffer (i+1)&nbsp;%&nbsp;2, so reads and writes never touch the same memory within a slab. The two things a barrier has to protect &mdash; don&rsquo;t overwrite what someone is still reading, don&rsquo;t read what isn&rsquo;t fully written &mdash; both line up on the single barrier between slabs. With one buffer those two hazards fall on the <em>same</em> memory inside one slab, which is exactly why it needs two.</figcaption>
</figure>

<figure>
  <div>{fig7()}</div>
  <figcaption><span class="n">7 &middot; warptiling</span>
  Per-thread work is untouched &mdash; every thread still owns an 8&times;8 tile, loads 8+8, does 64 FMAs. What changes is <em>which</em> thread gets <em>which</em> tile. Today a warp&rsquo;s 32 lanes lie along a 16&times;128 strip, so per dot step the warp needs the entire Bs row and only 16 floats of As. Reshaped to a 64&times;32 block, it needs 64 of As and 32 of Bs: 96 distinct floats instead of 144 for the same 2048 FMAs. Shared-memory bandwidth is what the kernel is spending now, so a third less traffic per warp is the lever.</figcaption>
<pre>const int warpId  = threadIdx.x / 32;   const int lane    = threadIdx.x % 32;
const int warpRow = warpId / 4;         const int warpCol = warpId % 4;    <span class="c">// 2 x 4 warps, 64 x 32 each</span>
const int laneRow = lane / 4;           const int laneCol = lane % 4;      <span class="c">// 8 x 4 lanes, 8 x 8 each</span>
const int threadRow = warpRow * 8 + laneRow;    <span class="c">// still 0..15  -> compute loop and epilogue unchanged</span>
const int threadCol = warpCol * 4 + laneCol;    <span class="c">// still 0..15</span></pre>
</figure>

<figure>
  <div>{fig8()}</div>
  <figcaption><span class="n">8 &middot; the bank conflict it removes</span>
  A warp&rsquo;s <code>LDS.128</code> is served eight lanes at a time. Each lane&rsquo;s b-fragment starts at byte <code>threadCol&nbsp;&times;&nbsp;32</code>, which lands on bank <code>(threadCol&nbsp;&times;&nbsp;8)&nbsp;%&nbsp;32</code> &mdash; so threadCol 0 and 4 share banks 0&ndash;3. Today those are lanes 0 and 4, reading different addresses: a 2-way conflict on every b load. Under the 8&times;4 lane grid, lanes 0 and 4 have the <em>same</em> threadCol, so they read the same address and the hardware broadcasts it. Same remap, second benefit.</figcaption>
</figure>
</main>
"""

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sgemm_tile_map.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out, len(HTML), "bytes")
