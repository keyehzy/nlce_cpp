#!/usr/bin/env python3
"""Test for a chiral Mott insulator (CMI) on the frustrated kagome lattice.

The test of tools/cmi_series.py, carried over to the kagome lattice: a CMI
exists iff the chiral susceptibility chi diverges at some x_chi while the
particle-hole gap is still open, x_chi < x_Delta.  nlce_run's hopping +t is
frustrated, x = t/U > 0.  Two translation-invariant chiral current patterns
are candidates (nlce_run --current, lattice.hpp):

    staggered   up and down triangles circulate oppositely (the
                sqrt3 x sqrt3 vector chirality; the triangular pattern
                restricted to the kagome bonds)
    uniform     every triangle circulates the same way (the q = 0 chirality)

The gap.  With three sites per cell, Hp and Hh are 3 x 3 Bloch matrices,
assembled from the pair-class amplitudes Hp_pairs, Hh_pairs.  At x^1 the
lowest band of each is the flat band (+t on the kagome), so the band minimum
is selected only at higher orders.  The flat-band energy has an exact series
at every k where it is non-degenerate at x^1 (Rayleigh-Schroedinger theory
of the matrix series, in Q(w), w^3 = 1); at Gamma it shares the
two-dimensional representation E with the touching dispersive band, whose
energy is the diagonal minus the off-diagonal element.  The particle and
the hole minima are located by scanning the zone with the truncated series
(--scan), and the gap is

    Delta = E_p(k_p) + E_h(k_h)

as one exact series.  Delta(0) of the unfrustrated side (the A1 band at
Gamma, i.e. tools/q0_series.py) fixes the Euler transform as in
cmi_series.py.

Usage: kagome_cmi.py SERIES.json [SERIES.json ...] [--scan] [--points]
"""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

import cmi_series as cs

# Kagome: the triangular lattice without the points with a and b even.
ANCHORS = {1: (1, 0), 2: (0, 1), 3: (1, 1)}


def residue(s):
    return s[0] % 2 + 2 * (s[1] % 2)


def cell(s):
    """Cell origin (even, even) of a site."""
    a = ANCHORS[residue(s)]
    return (s[0] - a[0], s[1] - a[1])


def pair_images(p, q):
    """Point-group images of the pair, translated so p is its anchor."""
    out = set()
    seen = {(p, q)}
    frontier = [(p, q)]
    while frontier:
        nxt = []
        for x, y in frontier:
            for g in (cs.ROTATE, cs.MIRROR):
                im = (g(x), g(y))
                if im not in seen:
                    seen.add(im)
                    nxt.append(im)
        frontier = nxt
    for x, y in seen:
        a = ANCHORS[residue(x)]
        sh = (a[0] - x[0], a[1] - x[1])
        out.add(((x[0] + sh[0], x[1] + sh[1]), (y[0] + sh[0], y[1] + sh[1])))
    return out


# Q(w), w = e^{2 pi i / 3}: a + b w with rational a, b.

class W:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fraction(a)
        self.b = Fraction(b)

    def __add__(self, o):
        o = lift(o)
        return W(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return W(-self.a, -self.b)

    def __sub__(self, o):
        return self + (-lift(o))

    def __rsub__(self, o):
        return lift(o) - self

    def __mul__(self, o):
        o = lift(o)
        return W(self.a * o.a - self.b * o.b, self.a * o.b + self.b * o.a - self.b * o.b)

    __rmul__ = __mul__

    def conj(self):
        return W(self.a - self.b, -self.b)

    def norm(self):
        return self.a * self.a - self.a * self.b + self.b * self.b

    def __truediv__(self, o):
        o = lift(o)
        n = o.norm()
        p = self * o.conj()
        return W(p.a / n, p.b / n)

    def __eq__(self, o):
        o = lift(o)
        return self.a == o.a and self.b == o.b

    def __bool__(self):
        return bool(self.a) or bool(self.b)

    def rational(self):
        if self.b:
            raise ValueError("not rational: %s + %s w" % (self.a, self.b))
        return self.a

    def __repr__(self):
        return "%s%+sw" % (self.a, self.b) if self.b else str(self.a)


def lift(x):
    return x if isinstance(x, W) else W(x)


def load(path):
    d = json.loads(Path(path).read_text())
    if d.get("lattice") != "kagome" or "pairs" not in d:
        raise SystemExit("%s: not a kagome series with pair classes" % path)
    ng = d["order_gap"]
    ser = lambda xs: [Fraction(x) for x in xs]
    # amp[r] = list of (s, cell displacement, Hp series, Hh series): from
    # site q on sublattice s to the anchor of sublattice r.
    amp = {r: [] for r in ANCHORS}
    for key, count, hp, hh in zip(d["pairs"], d["pair_counts"], d["Hp_pairs"], d["Hh_pairs"]):
        p, q = (key[0], key[1]), (key[2], key[3])
        images = pair_images(p, q)
        if len(images) != count:
            raise SystemExit("pair class %s: %d images, %d expected" % (key, len(images), count))
        for x, y in images:
            cx, cy = cell(x), cell(y)
            amp[residue(x)].append((residue(y), (cy[0] - cx[0], cy[1] - cx[1]), ser(hp), ser(hh)))
    gap0 = [Fraction(0)] * (ng + 1)
    for size, hp, hh in zip(d["orbit_sizes"], d["Hp"], d["Hh"]):
        for k in range(ng + 1):
            gap0[k] += size * (Fraction(hp[k]) + Fraction(hh[k]))
    out = {"v": d["v"], "nsites": d["nsites"], "ng": ng, "amp": amp, "gap0": gap0}
    if "chi" in d:
        out["chi"] = ser(d["chi"])
        out["current"] = d["current"]
    return out


# Bloch matrices.  The phase of a hop between cells displaced by R (even
# coordinates) is e^{i (t1 R_a + t2 R_b)}, t_i = k.e_i.  At the points below
# the phases are cube roots of unity or signs.

POINTS = {
    # (t1, t2) in units of pi/6; k.(2 e1), k.(2 e2) are (0, 0), (-2pi/3, 2pi/3)
    # and (pi, 0)
    "Gamma": (0, 0),
    "K": (-2, 2),
    "M": (3, 0),
}

SIXTH = W(1, 1)  # e^{i pi/3} = 1 + w


def exact_phase(point, R):
    t1, t2 = POINTS[point]
    m = t1 * R[0] + t2 * R[1]  # phase e^{i pi m / 6}, m even
    assert m % 2 == 0
    out = W(1)
    for _ in range((m // 2) % 6):
        out = out * SIXTH
    return out


def bloch_exact(s, point, which):
    """H(k)[order][r][c] in Q(w), r, c = sublattice - 1."""
    ng = s["ng"]
    H = [[[W() for _ in range(3)] for _ in range(3)] for _ in range(ng + 1)]
    for r, hops in s["amp"].items():
        for c, R, hp, hh in hops:
            ph = exact_phase(point, R)
            xs = hp if which == "p" else hh
            for k in range(ng + 1):
                if xs[k]:
                    H[k][r - 1][c - 1] = H[k][r - 1][c - 1] + ph * xs[k]
    return H


def bloch_numeric(s, which, t1, t2, order):
    """Numeric Bloch matrices through x^order at k = (t1, t2), as complex arrays."""
    H = np.zeros((order + 1, 3, 3), dtype=complex)
    for r, hops in s["amp"].items():
        for c, R, hp, hh in hops:
            ph = np.exp(1j * (t1 * R[0] + t2 * R[1]))
            xs = hp if which == "p" else hh
            for k in range(order + 1):
                H[k, r - 1, c - 1] += ph * float(xs[k])
    return H


def matvec(M, v):
    return [sum((M[i][j] * v[j] for j in range(3)), W()) for i in range(3)]


def inverse3(M):
    A = [[lift(M[i][j]) for j in range(3)] + [W(int(i == j)) for j in range(3)] for i in range(3)]
    for i in range(3):
        p = next(r for r in range(i, 3) if A[r][i])
        A[i], A[p] = A[p], A[i]
        piv = A[i][i]
        A[i] = [x / piv for x in A[i]]
        for r in range(3):
            if r != i and A[r][i]:
                f = A[r][i]
                A[r] = [x - f * y for x, y in zip(A[r], A[i])]
    return [row[3:] for row in A]


def null_vector(M):
    """A nonzero v with M v = 0 for a rank-2 3 x 3 matrix."""
    rows = [[lift(x) for x in row] for row in M]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = rows[i], rows[j]
        v = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        if any(v):
            return v
    raise ValueError("rank below 2")


def band_series(H, lam0):
    """Rayleigh-Schroedinger series of the eigenvalue of sum_k H[k+1] x^k that
    starts at the simple eigenvalue lam0 of H[1]; H[0] must be a multiple of
    the identity.  Returns the eigenvalue of sum_k H[k] x^k."""
    n = len(H) - 1
    c0 = H[0][0][0]
    for i in range(3):
        for j in range(3):
            if H[0][i][j] != (c0 if i == j else 0):
                raise ValueError("order 0 is not a multiple of the identity")
    M = H[1:]
    A = [[M[0][i][j] - (lam0 if i == j else 0) for j in range(3)] for i in range(3)]
    r = null_vector(A)
    l = null_vector([[A[j][i] for j in range(3)] for i in range(3)])
    lr = sum((l[i] * r[i] for i in range(3)), W())
    l = [x / lr for x in l]
    B = inverse3([[A[i][j] + r[i] * l[j] for j in range(3)] for i in range(3)])
    psi = [r]
    lam = [lift(lam0)]
    for m in range(1, n):
        acc = [W(), W(), W()]
        for k in range(1, m + 1):
            mv = matvec(M[k], psi[m - k])
            acc = [a + b for a, b in zip(acc, mv)]
        lm = sum((l[i] * acc[i] for i in range(3)), W())
        lam.append(lm)
        rhs = [W(), W(), W()]
        for k in range(1, m + 1):
            mv = matvec(M[k], psi[m - k])
            rhs = [x + lam[k] * y - z for x, y, z in zip(rhs, psi[m - k], mv)]
        nxt = matvec(B, rhs)
        if sum((l[i] * nxt[i] for i in range(3)), W()):
            raise ValueError("reduced resolvent left the complement")
        psi.append(nxt)
    return [c0.rational()] + [x.rational() for x in lam]


def flat_band(s, point, which):
    """Exact series of the flat-band energy at a high-symmetry point."""
    H = bloch_exact(s, point, which)
    if point == "Gamma":
        # E representation: diagonal minus off-diagonal.
        return [(H[k][0][0] - H[k][0][1]).rational() for k in range(len(H))]
    lam0 = min(np.linalg.eigvals(np.array([[complex(float(x.a) - 0.5 * float(x.b), math.sqrt(3) / 2 * float(x.b))
                                             for x in row] for row in H[1]])).real)
    return band_series(H, Fraction(round(lam0)))


def numeric_band(s, which, t1, t2, x, order):
    """Lowest eigenvalue of the series truncated at x^order."""
    H = bloch_numeric(s, which, t1, t2, order)
    M = sum(H[k] * x ** k for k in range(order + 1))
    return min(np.linalg.eigvals(M).real)


def scan_zone(s, which, order, x, n=60):
    """(t1, t2, energy) of the lowest band over the zone at small x."""
    best = None
    grid = []
    for i in range(n):
        for j in range(n):
            t1, t2 = math.pi * i / n, math.pi * j / n
            e = numeric_band(s, which, t1, t2, x, order)
            grid.append((e, t1, t2))
            if best is None or e < best[0]:
                best = (e, t1, t2)
    return best, grid


def point_name(t1, t2):
    """Name of the high-symmetry point equivalent to (t1, t2), if any."""
    def same(a, b):
        d = (a - b) / math.pi
        return abs(d - round(d)) < 1e-9
    for name, (p1, p2) in POINTS.items():
        for g1, g2 in kagome_star(p1 * math.pi / 6, p2 * math.pi / 6):
            if same(t1, g1) and same(t2, g2):
                return name
    return None


def kagome_star(t1, t2):
    """Point-group images of k = (t1, t2), with t_i = k.e_i."""
    # A rotation of real space by 60 degrees sends e1 -> e2, e2 -> e2 - e1.
    out = set()
    pts = [(t1, t2)]
    for _ in range(6):
        a, b = pts[-1]
        pts.append((b, b - a))
    for a, b in pts:
        out.add((a, b))
        out.add((a, a - b))  # mirror e1 -> e1, e2 -> e1 - e2
    return out


def gap_series(s, kp, kh):
    p = flat_band(s, kp, "p")
    h = flat_band(s, kh, "h")
    return [a + b for a, b in zip(p, h)], p, h


def analyse(s, args):
    ng = s["ng"]
    print("V/U = %s, s = %d (gap through x^%d%s)" % (
        s["v"], s["nsites"], ng, ", chi (%s) through x^%d" % (s["current"], len(s["chi"]) - 1) if "chi" in s else ""))
    bands = {}
    for which in ("p", "h"):
        bands[which] = {pt: flat_band(s, pt, which) for pt in POINTS}
    if args.points:
        for which, title in (("p", "particle"), ("h", "hole")):
            for pt, ser in bands[which].items():
                print("  %-8s %-5s " % (title, pt) + "  ".join("%.6g" % float(c) for c in ser))

    # Which point is lowest for small x > 0: the series that is smaller at
    # the first order where they differ.  The zone scan checks that no other
    # k lies lower.
    choice = {}
    for which in ("p", "h"):
        ser = bands[which]
        first = lambda pt: next((k for k in range(ng + 1) if any(ser[q][k] != ser[pt][k] for q in ser)), ng + 1)
        k0 = min(first(pt) for pt in ser)
        low = min(ser, key=lambda pt: ser[pt][k0] if k0 <= ng else 0)
        choice[which] = low
        x = 0.02
        (e, t1, t2), _ = scan_zone(s, which, ng, x, n=48 if args.scan else 18)
        e_low = numeric_band(s, which, *[c * math.pi / 6 for c in POINTS[low]], x, ng)
        print("  %s band lowest at %s (bands differ from x^%d); zone scan at x = %.2f: minimum at (%.3f, %.3f) pi,"
              " %.2e below %s" % ("particle" if which == "p" else "hole    ", low, k0, x, t1 / math.pi, t2 / math.pi,
                                  e_low - e, low))
    gap, _, _ = gap_series(s, choice["p"], choice["h"])
    print("  gap = E_p(%s) + E_h(%s): %s" % (choice["p"], choice["h"], "  ".join("%.6g" % float(c) for c in gap)))

    alt = [c * (-1) ** k for k, c in enumerate(s["gap0"])]
    neg = cs.poles(alt, 0.005, 0.2)
    a = Fraction(float(np.median([p[0] for p in neg]))).limit_denominator(1000)
    to_x = lambda z: z / (1 - z / a)
    to_z = lambda x: x / (1 + x / a)
    print("  unfrustrated MI-SF singularity        x = -%.4f  (%d approximants)" % (float(a), len(neg)))

    gp = cs.poles(cs.euler(gap, a), 0.005, 0.95 * float(a))
    xs = np.array([float(to_x(Fraction(p[0]))) for p in gp])
    x_delta = None
    if len(xs):
        x_delta, err = float(np.median(xs)), float(xs.std())
        znu = -float(np.median([p[1] for p in gp]))
        print("  gap closes                            x_Delta = %.4f +- %.4f  (z nu = %.2f, %d approximants)" % (
            x_delta, err, znu, len(xs)))
        if args.verbose:
            print("    poles in x: " + " ".join("%.4f" % x for x in sorted(xs)))
    else:
        print("  gap: no real dlog Pade pole for 0 < x < %.1f" % float(to_x(Fraction(0.95) * a)))
    for x in (0.1, 0.2, 0.3, 0.4):
        vals = [cs.value(P, float(to_z(Fraction(x)))) / cs.value(Q, float(to_z(Fraction(x))))
                for P, Q in (r for r in (cs.pade(cs.euler(gap, a), L, M) for L, M in cs.approximants(len(gap) + 1))
                             if r is not None)]
        print("    gap at x = %.1f: %s" % (x, " ".join("%.3f" % v for v in vals)))
    if "chi" not in s:
        print()
        return

    chi = cs.euler(s["chi"], a)
    cp = cs.poles(chi, 0.005, 0.95 * float(a))
    if cp:
        cx = np.array([float(to_x(Fraction(p[0]))) for p in cp])
        print("  chi unbiased dlog Pade                x_chi = %.4f +- %.4f  (gamma = %.2f, %d approximants)" % (
            float(np.median(cx)), float(cx.std()), float(np.median([p[1] for p in cp])), len(cx)))
        if args.verbose:
            print("    poles in x: " + " ".join("%.4f(%.2f)" % (float(to_x(Fraction(p[0]))), p[1]) for p in cp))
    else:
        print("  chi unbiased dlog Pade                no real pole")
    for xb in ([x_delta] if x_delta else []) + args.bias:
        g = cs.biased_gamma(chi, to_z(Fraction(xb).limit_denominator(10 ** 6)))
        print("  chi biased at x_c = %.4f              gamma = %.3f +- %.3f" % (xb, g.mean(), g.std()))
    hi = x_delta if x_delta else args.xmax
    grid = [Fraction(n, 2000) for n in range(20, int(2000 * hi) + 1)]
    rows = []
    for xc in grid:
        g = cs.biased_gamma(chi, to_z(xc))
        rows.append((float(xc), g.mean(), g.std()))
    best = min(rows, key=lambda r: r[2])
    print("  chi approximants agree best at        x_c = %.4f  (gamma = %.3f +- %.3f)" % best)
    cross = [r for r, q in zip(rows, rows[1:]) if (r[1] - cs.GAMMA_ISING) * (q[1] - cs.GAMMA_ISING) <= 0]
    if cross:
        print("  Ising gamma at                        " + ", ".join(
            "x_c = %.4f (+- %.3f)" % (r[0], r[2]) for r in cross))
    else:
        print("  Ising gamma                           nowhere in [%.3f, %.3f]" % (rows[0][0], rows[-1][0]))
    if args.scan:
        for xc, g, sd in rows[::20]:
            print("    x_c = %.4f  gamma = %7.3f +- %.3f" % (xc, g, sd))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--scan", action="store_true", help="finer zone scan and the biased exponent of chi against x_c")
    ap.add_argument("--points", action="store_true", help="print the flat-band series at Gamma, K and M")
    ap.add_argument("--verbose", action="store_true", help="list the individual Pade poles")
    ap.add_argument("--bias", type=float, action="append", default=[], help="also bias chi at this x_c")
    ap.add_argument("--xmax", type=float, default=0.5, help="end of the x_c scan when the gap does not close")
    args = ap.parse_args()
    for path in args.json:
        analyse(load(path), args)


if __name__ == "__main__":
    main()
