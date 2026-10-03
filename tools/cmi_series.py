#!/usr/bin/env python3
"""Test for a chiral Mott insulator (CMI) on the frustrated triangular lattice.

A CMI is a Mott insulator with spontaneous staggered loop currents.  Coming
from the Mott side, it appears where the chiral susceptibility chi diverges
(a 3D Ising transition, gamma = 1.237) while the charge gap is still open.
Without a CMI, chi diverges only where the gap closes, at the transition to
the (chiral) superfluid.  The test therefore compares two singularities of
the nlce_run series, in its +t convention where x = t/U > 0 is frustrated:

    Delta(K) = sum_d (Hp + Hh)[d] e^{iK.d}, the particle-hole gap at the band
               minimum K, vanishing at x_Delta
    chi        the staggered-current susceptibility, diverging at x_chi

and a CMI exists iff x_chi < x_Delta.

Every series is dominated by the Mott-superfluid transition of the
unfrustrated model at x = -a, a ~ 0.03-0.04 (Delta(q=0) closing for -t
hopping).  The Euler transform x = z / (1 - z/a) sends it to z = infinity;
a is located by dlog Pade of Delta(0) on the negative axis.  Then

  * x_Delta: real dlog Pade poles of Delta(K) in z (unbiased);
  * chi: biased dlog Pade, gamma(x_c) = lim (x_c - x) d ln chi / dx at x_c,
    for a range of x_c.  The approximants agree near the true singularity.
    Placing the singularity at x_Delta tests "no CMI"; asking where
    gamma(x_c) = 1.237 with x_c < x_Delta tests "CMI".

Usage: cmi_series.py SERIES.json [SERIES.json ...] [--scan]
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

GAMMA_ISING = 1.2372

ROTATE = lambda s: (-s[1], s[0] + s[1])
MIRROR = lambda s: (s[0] + s[1], -s[1])


def orbit(d):
    """D6 images of a displacement (a, b) at a e1 + b e2."""
    seen, frontier = {tuple(d)}, [tuple(d)]
    while frontier:
        nxt = []
        for s in frontier:
            for g in (ROTATE, MIRROR):
                t = g(s)
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    return seen


def cos_k(s):
    """cos(K.d) at the zone corner K, where e^{iK.d} = w^(a - b), w^3 = 1."""
    return Fraction(1) if (s[0] - s[1]) % 3 == 0 else Fraction(-1, 2)


def load(path):
    d = json.loads(Path(path).read_text())
    if "chi" not in d:
        raise SystemExit("%s has no chi series (triangular lattice only)" % path)
    ng = d["order_gap"]
    gap_k = [Fraction(0)] * (ng + 1)
    gap_0 = [Fraction(0)] * (ng + 1)
    for disp, size, hp, hh in zip(d["displacements"], d["orbit_sizes"], d["Hp"], d["Hh"]):
        images = orbit(disp)
        assert len(images) == size
        weight = sum(cos_k(s) for s in images)
        for k in range(ng + 1):
            h = Fraction(hp[k]) + Fraction(hh[k])
            gap_k[k] += weight * h
            gap_0[k] += size * h
    return {"v": d["v"], "nsites": d["nsites"], "gapK": gap_k, "gap0": gap_0,
            "chi": [Fraction(x) for x in d["chi"]]}


# Exact series arithmetic and Pade approximants.

def mul(a, b, n):
    return [sum(a[j] * b[k - j] for j in range(k + 1) if j < len(a) and k - j < len(b)) for k in range(n)]


def inverse(a, n):
    b = [Fraction(0)] * n
    b[0] = 1 / a[0]
    for k in range(1, n):
        b[k] = -sum(a[j] * b[k - j] for j in range(1, min(k, len(a) - 1) + 1)) / a[0]
    return b


def dlog(c):
    n = len(c) - 1
    return mul([k * c[k] for k in range(1, len(c))], inverse(c, n), n)


def compose(c, x, n):
    """c(x(z)) through z^(n-1), x(0) = 0."""
    out = [Fraction(0)] * n
    power = [Fraction(1)] + [Fraction(0)] * (n - 1)
    for k in range(min(len(c), n)):
        out = [o + c[k] * p for o, p in zip(out, power)]
        power = mul(power, x, n)
    return out


def euler(c, a):
    """c(x(z)) with x = z / (1 - z/a)."""
    n = len(c)
    return compose(c, [Fraction(0)] + [a ** (1 - k) for k in range(1, n)], n)


def solve(rows, rhs):
    m = [r[:] + [b] for r, b in zip(rows, rhs)]
    n = len(m)
    for i in range(n):
        p = next((r for r in range(i, n) if m[r][i] != 0), None)
        if p is None:
            return None
        m[i], m[p] = m[p], m[i]
        for r in range(n):
            if r != i and m[r][i] != 0:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * y for x, y in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def pade(c, L, M):
    """[L/M] Pade of the series c as (P, Q), Q[0] = 1, or None if singular."""
    at = lambda k: c[k] if 0 <= k < len(c) else Fraction(0)
    q = solve([[at(L + i - j) for j in range(1, M + 1)] for i in range(1, M + 1)],
              [-at(L + i) for i in range(1, M + 1)])
    if q is None:
        return None
    Q = [Fraction(1)] + q
    return [sum(Q[j] * at(k - j) for j in range(min(k, M) + 1)) for k in range(L + 1)], Q


def value(p, x):
    return sum(float(a) * x ** k for k, a in enumerate(p))


def roots(p):
    p = [float(a) for a in p]
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return np.roots(p[::-1]) if len(p) > 1 else np.array([])


def approximants(n):
    """Near-diagonal [L/M], M >= 2, from the two longest series of n terms."""
    return [(L, t - L) for t in (n - 2, n - 1) for L in range(1, t - 1) if abs(2 * L - t) <= 2]


def poles(c, lo, hi):
    """Smallest real dlog-Pade pole in (lo, hi) per approximant, skipping
    pole-zero defects; (x_c, exponent) with c ~ (x_c - x)^(-exponent)."""
    d = dlog(c)
    out = []
    for L, M in approximants(len(d)):
        r = pade(d, L, M)
        if r is None:
            continue
        P, Q = r
        zeros = roots(P)
        dQ = [k * Q[k] for k in range(1, len(Q))]
        found = []
        for z in roots(Q):
            if abs(z.imag) > 1e-9 or not lo < z.real < hi:
                continue
            z = z.real
            residue = value(P, z) / value(dQ, z)
            if abs(residue) > 0.02 and not any(abs(w - z) < 2e-3 * z for w in zeros):
                found.append((z, -residue))
        if found:
            out.append(min(found))
    return out


def biased_gamma(c, zc):
    """(z_c - z) d ln c/dz at z_c from each approximant."""
    d = dlog(c)
    g = [zc * d[k] - (d[k - 1] if k else 0) for k in range(len(d))]
    out = []
    for L, M in approximants(len(g)):
        r = pade(g, L, M)
        if r is not None:
            out.append(value(r[0], float(zc)) / value(r[1], float(zc)))
    return np.array(out)


def analyse(s, scan):
    alt = [c * (-1) ** k for k, c in enumerate(s["gap0"])]
    neg = poles(alt, 0.005, 0.2)
    a = Fraction(float(np.median([p[0] for p in neg]))).limit_denominator(1000)
    to_x = lambda z: z / (1 - z / a)
    to_z = lambda x: x / (1 + x / a)

    gap = poles(euler(s["gapK"], a), 0.005, 0.95 * float(a))
    xs = np.array([float(to_x(Fraction(p[0]))) for p in gap])
    x_delta, err = float(np.median(xs)), float(xs.std())
    znu = -float(np.median([p[1] for p in gap]))

    chi = euler(s["chi"], a)
    at_gap = biased_gamma(chi, to_z(Fraction(x_delta).limit_denominator(10 ** 6)))
    print("V/U = %s, s = %d (Delta through x^%d, chi through x^%d)" % (
        s["v"], s["nsites"], len(s["gapK"]) - 1, len(s["chi"]) - 1))
    print("  unfrustrated MI-SF singularity        x = -%.4f" % float(a))
    print("  Delta(K) closes                       x_Delta = %.4f +- %.4f  (z nu = %.2f, %d approximants)" % (
        x_delta, err, znu, len(xs)))
    print("  chi biased at x_Delta                 gamma = %.3f +- %.3f" % (at_gap.mean(), at_gap.std()))

    grid = [Fraction(n, 4000) for n in range(int(4000 * x_delta) - 80, int(4000 * x_delta) + 81)]
    rows = []
    for xc in grid:
        g = biased_gamma(chi, to_z(xc))
        rows.append((float(xc), g.mean(), g.std()))
    best = min(rows, key=lambda r: r[2])
    print("  chi approximants agree best at        x_c = %.4f  (gamma = %.3f +- %.3f)" % best)
    # Below x_Delta, the nearest x_c at which gamma(x_c) crosses the Ising value.
    below = [r for r in rows if r[0] < x_delta][::-1]
    cross = next((r for r, q in zip(below, below[1:]) if (r[1] - GAMMA_ISING) * (q[1] - GAMMA_ISING) <= 0), None)
    if cross:
        print("  Ising gamma below x_Delta at          x_c = %.4f  (gamma = %.3f +- %.3f)" % cross)
    else:
        print("  Ising gamma below x_Delta             nowhere in [%.4f, %.4f)" % (below[-1][0], x_delta))
    if scan:
        for xc, g, sd in rows[::8]:
            print("    x_c = %.4f  gamma = %7.3f +- %.3f%s" % (xc, g, sd, "   <- x_Delta" if abs(xc - x_delta) < 1e-3 else ""))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--scan", action="store_true", help="print the biased exponent of chi against x_c")
    args = ap.parse_args()
    for path in args.json:
        analyse(load(path), args.scan)


if __name__ == "__main__":
    main()
