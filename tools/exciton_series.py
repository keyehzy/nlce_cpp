#!/usr/bin/env python3
"""Series of the chiral doublon-holon exciton energy on the triangular lattice.

Reads `nlce_run run --dh 1` output: the one-doublon and one-holon effective
Hamiltonians Hp, Hh (through x^(N-1)) and the doublon-holon interaction Idh
(through x^(N-2)), and forms the two-particle effective Hamiltonian at total
momentum zero in the relative coordinate r = r_doublon - r_holon != 0:

    H(r' <- r) = Hp[r' - r] + Hh[r - r'] + Idh[(r', r)].

At x = 0 an adjacent pair has energy 1 - V and every other pair 1.  The chiral
(staggered-current) channel holds one adjacent combination, chi(d) = sigma(d),
the sign of the staggered current on the bond d, which alternates around the
hexagon; it is the state the current operator creates from the Mott state.
For V > 0 it is non-degenerate within its symmetry sector, so its energy E_ex
has an ordinary Rayleigh-Schroedinger series, exact through x^(N-2).  The
vacuum is even under time reversal and the chiral state odd, so they never mix.

A chiral Mott insulator appears where E_ex reaches zero while the
particle-hole gap Delta(K) (tools/cmi_series.py) is still open.  Printed: the
series of E_ex, of Delta(K), and of the binding Delta(K) - E_ex; the zeros of
E_ex (near-diagonal Pade) and of Delta(K) (dlog Pade, as in cmi_series.py),
each after an Euler transform x = z / (1 - z/a) that removes the series' own
nearest singularity on the unfrustrated side x = -a; and E_ex where Delta(K)
closes.  E_ex reaches its singularity much closer to the origin than Delta
does, near x = -V/24, where the exciton meets the continuum at Gamma for -t
hopping; that limits how far its series can be continued.  --solve instead
resums each amplitude and solves the two-body problem at each x.

Usage: exciton_series.py SERIES.json [SERIES.json ...] [--solve]
"""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import cmi_series as cs

NEIGHBOURS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def sigma(d):
    """Staggered current sign of the bond from the origin to d (lattice.cpp)."""
    return 1 if (2 * d[0] + d[1]) % 3 == 1 else -1


def hop_distance(r):
    a, b = r
    return max(abs(a), abs(b), abs(a + b))


def group_images(r2, r):
    """Simultaneous point-group images of the pair (r2, r)."""
    seen = {(tuple(r2), tuple(r))}
    frontier = list(seen)
    while frontier:
        nxt = []
        for x, y in frontier:
            for g in (cs.ROTATE, cs.MIRROR):
                im = (g(x), g(y))
                if im not in seen:
                    seen.add(im)
                    nxt.append(im)
        frontier = nxt
    return seen


def load(path):
    d = json.loads(Path(path).read_text())
    if "Idh" not in d:
        raise SystemExit("%s has no Idh series; run nlce_run with --dh 1" % path)
    n = d["order_dh"]
    series = lambda xs: [Fraction(x) for x in xs[: n + 1]]
    hp, hh = {}, {}
    for disp, a, b in zip(d["displacements"], d["Hp"], d["Hh"]):
        for s in cs.orbit(disp):
            hp[s] = series(a)
            hh[s] = series(b)
    idh = {}
    for key, xs in zip(d["dh_keys"], d["Idh"]):
        xs = series(xs)
        if not any(xs):
            continue
        for im in group_images(key[:2], key[2:]):
            idh[im] = xs
    gaps = cs.load(path)
    return {"v": Fraction(d["v"]), "nsites": d["nsites"], "n": n, "Hp": hp, "Hh": hh, "Idh": idh,
            "gapK": gaps["gapK"], "gap0": gaps["gap0"]}


def hamiltonian(s, radius):
    """H[(r2, r)] series for 0 < |r|, |r2| <= radius (hop distance)."""
    n = s["n"]
    zero = [Fraction(0)] * (n + 1)
    sites = [(a, b) for a in range(-radius, radius + 1) for b in range(-radius, radius + 1)
             if 0 < hop_distance((a, b)) <= radius]
    index = set(sites)
    H = {}

    def add(key, xs):
        cur = H.setdefault(key, zero[:])
        for k in range(n + 1):
            cur[k] += xs[k]

    for r in sites:
        for d, xs in s["Hp"].items():
            r2 = (r[0] + d[0], r[1] + d[1])
            if r2 in index:
                add((r2, r), xs)
        for e, xs in s["Hh"].items():
            r2 = (r[0] - e[0], r[1] - e[1])
            if r2 in index:
                add((r2, r), xs)
    for (r2, r), xs in s["Idh"].items():
        if r2 in index and r in index:
            add((r2, r), xs)
    return sites, H


def exciton_energy(s):
    """Rayleigh-Schroedinger series of the chiral exciton energy."""
    n = s["n"]
    v = s["v"]
    if v <= 0:
        raise SystemExit("the chiral exciton is degenerate with the continuum at V = 0")
    sites, H = hamiltonian(s, n + 2)
    diag0 = {r: H.get((r, r), [Fraction(0)])[0] for r in sites}
    for (r2, r), xs in H.items():
        if r2 != r and xs[0] != 0:
            raise SystemExit("off-diagonal amplitude at x^0")
    shell = {d: Fraction(sigma(d)) for d in NEIGHBOURS}
    e0 = diag0[NEIGHBOURS[0]]
    assert all(diag0[d] == e0 for d in NEIGHBOURS) and e0 == 1 - v
    # W_m: columns of the order-m perturbation, as {r: [(r2, w)]}.
    cols = [dict() for _ in range(n + 1)]
    for (r2, r), xs in H.items():
        for m in range(1, n + 1):
            if xs[m]:
                cols[m].setdefault(r, []).append((r2, xs[m]))

    def apply(m, vec):
        out = {}
        for r, c in vec.items():
            for r2, w in cols[m].get(r, ()):
                out[r2] = out.get(r2, 0) + w * c
        return out

    psi = [dict(shell)]
    energy = [e0]
    for k in range(1, n + 1):
        ek = Fraction(0)
        for m in range(1, k + 1):
            w = apply(m, psi[k - m])
            ek += sum(w.get(d, 0) * c for d, c in shell.items()) / 6
        energy.append(ek)
        if k == n:
            break
        rhs = {}
        for m in range(1, k + 1):
            for r, c in apply(m, psi[k - m]).items():
                rhs[r] = rhs.get(r, 0) + c
            for r, c in psi[k - m].items():
                rhs[r] = rhs.get(r, 0) - energy[m] * c
        nxt = {}
        for r, c in rhs.items():
            if r in shell:
                continue
            if c:
                nxt[r] = c / (e0 - diag0[r])
        # Within the adjacent shell only chi itself is degenerate; the rest of
        # the shell lies in other symmetry sectors and must stay empty.
        along = sum(rhs.get(d, 0) * c for d, c in shell.items()) / 6
        rest = {d: rhs.get(d, 0) - along * c for d, c in shell.items()}
        if along or any(rest.values()):
            raise SystemExit("chiral sector not closed at order %d" % k)
        psi.append(nxt)
    return energy


def negative_singularity(c):
    """Nearest real singularity on the unfrustrated side x < 0, from dlog Pade
    of c(-x); weak (small residue) singularities count."""
    d = cs.dlog([x * (-1) ** k for k, x in enumerate(c)])
    found = []
    for L, M in cs.approximants(len(d)):
        r = cs.pade(d, L, M)
        if r is None:
            continue
        zs = [z.real for z in cs.roots(r[1]) if abs(z.imag) < 1e-9 and 0.003 < z.real < 0.2]
        if zs:
            found.append(min(zs))
    return Fraction(float(np.median(found))).limit_denominator(10000)


def approximants(e):
    for L in range(1, len(e)):
        for M in range(0, len(e) - L):
            if L + M >= len(e) - 2 and abs(L - M) <= 2:
                r = cs.pade(e, L, M)
                if r is not None:
                    yield r


def zeros(c, a, lo=0.005, hi=0.3):
    """Smallest positive real zero of each near-diagonal Pade of the
    Euler-transformed series, away from pole-zero pairs."""
    out = []
    for P, Q in approximants(cs.euler(c, a)):
        pz = [z.real for z in cs.roots(P) if abs(z.imag) < 1e-9 and z.real > 0]
        qz = [z.real for z in cs.roots(Q) if abs(z.imag) < 1e-9 and z.real > 0]
        pz = [z for z in pz if not any(abs(z - q) < 1e-3 * z for q in qz)]
        xs = [float(z / (1 - z / a)) for z in pz if z < 0.95 * float(a)]
        xs = [x for x in xs if lo < x < hi]
        if xs:
            out.append(min(xs))
    return out


def values(c, a, x):
    """The near-diagonal Pade values of c at x after the Euler transform."""
    z = x / (1 + x / float(a))
    return [cs.value(P, z) / cs.value(Q, z) for P, Q in approximants(cs.euler(c, a))]


def summary(xs):
    return "%.4f (%.4f..%.4f, %d)" % (np.median(xs), min(xs), max(xs), len(xs)) if xs else "-"


# Non-perturbative two-body solution (--solve).  The amplitudes Hp, Hh and Idh each
# converge out to the vacuum singularity of the unfrustrated model (x ~ -0.03),
# while E_ex already ends where the exciton unbinds for -t hopping, near -V/24.
# So each amplitude is resummed (Euler transform past -a, then [L/M] Pade) and
# the bound state found from the resummed two-body problem at each x:
#   (H_kin + U) psi = E psi,  H_kin = Delta(k) in the relative momentum,
# whose bound states solve det(1 + G(E) U) = 0 on the support S of U, with
# G(E) = (Delta - E)^-1, restricted to the chiral representation (odd under
# 60-degree rotations, even under the mirrors through bonds).  There psi
# vanishes at r = 0, so the hard core costs nothing.  Resumming amplitude by
# amplitude is less accurate than resumming Delta(K) itself: at s = 10 it
# reproduces the series for x <~ 0.05 but places the gap closing too far out,
# so its output near the transition is qualitative.

def euler_matrix(n, a):
    """T[j][m]: coefficient of z^j in (z / (1 - z/a))^m, j, m <= n."""
    T = np.zeros((n + 1, n + 1))
    T[0, 0] = 1
    for m in range(1, n + 1):
        for j in range(m, n + 1):
            T[j, m] = math.comb(j - 1, m - 1) * a ** (m - j)
    return T


def pade_values(c, L, M, z):
    """[L/M] Pade of each row of c (batch x terms) at z."""
    B = c.shape[0]
    q = np.ones((B, M + 1))
    if M:
        A = np.stack([np.stack([c[:, L + i - j] if L + i - j >= 0 else np.zeros(B) for j in range(1, M + 1)], -1)
                      for i in range(1, M + 1)], 1)
        rhs = -np.stack([c[:, L + i] for i in range(1, M + 1)], -1)
        # Amplitudes that start late make A singular; the pseudo-inverse then
        # drops the empty directions instead of failing.
        q[:, 1:] = (np.linalg.pinv(A, rcond=1e-13) @ rhs[..., None])[..., 0]
    p = np.stack([sum(q[:, j] * c[:, k - j] for j in range(0, min(k, M) + 1)) for k in range(L + 1)], -1)
    return (p @ (z ** np.arange(L + 1))) / (q @ (z ** np.arange(M + 1)))


ROT60 = lambda r: (-r[1], r[0] + r[1])
MIRROR_BOND = lambda r: (r[0] + r[1], -r[1])


def chiral_basis(sites):
    """Orthonormal basis of the chiral representation on a symmetric site set."""
    index = {r: i for i, r in enumerate(sites)}
    P = np.zeros((len(sites), len(sites)))
    for r in sites:
        x = r
        for k in range(6):
            P[index[x], index[r]] += (-1) ** k
            P[index[MIRROR_BOND(x)], index[r]] += (-1) ** k
            x = ROT60(x)
    P /= 12
    w, v = np.linalg.eigh((P + P.T) / 2)
    return v[:, w > 0.5]


class TwoBody:
    def __init__(self, s, a, nk=240):
        self.n, self.ng = s["n"], len(s["gapK"]) - 1
        self.a = float(a)
        kin = {}
        for d in set(s["Hp"]) | set(s["Hh"]):
            xs = [float(p + h) for p, h in zip(s["Hp"].get(d, [0] * (self.ng + 1)), s["Hh"].get(d, [0] * (self.ng + 1)))]
            kin[d] = xs
        self.ng = len(next(iter(kin.values()))) - 1
        k = 2 * np.pi * np.arange(nk) / nk
        k1, k2 = np.meshgrid(k, k, indexing="ij")
        self.nk = nk
        # Delta(k) is assembled from the resummed real-space amplitudes, so it
        # is smooth in k whatever the approximants do.
        self.disp = list(kin)
        self.phases = np.stack([np.cos(da * k1 + db * k2).ravel() for da, db in self.disp])
        self.kin_series = np.array([kin[d] for d in self.disp]) @ euler_matrix(self.ng, self.a).T
        self.keys = [key for key in s["Idh"]]
        u = np.array([[float(c) for c in s["Idh"][key]] for key in self.keys])
        self.u_series = u @ euler_matrix(self.n, self.a).T
        self.support = sorted({key[0] for key in self.keys} | {key[1] for key in self.keys} | set(NEIGHBOURS))
        self.basis = chiral_basis(self.support)
        pos = {r: i for i, r in enumerate(self.support)}
        self.rows = np.array([pos[key[0]] for key in self.keys])
        self.cols = np.array([pos[key[1]] for key in self.keys])
        S = np.array(self.support)
        self.diff = (S[:, None, :] - S[None, :, :]) % nk

    @staticmethod
    def resum(series, x, a, total):
        """Median over the near-diagonal [L/M], L + M = total or total - 1,
        |L - M| <= 2, M >= 1, of each row's Pade value: every single
        approximant has spurious poles at scattered k points or amplitudes
        (at s = 10, from 24 to 1746 of 240^2 k points at x = 0.06), the
        median of the five has none."""
        z = x / (1 + x / a)
        vals = [pade_values(series[:, : t + 1], L, t - L, z) for t in (total, total - 1) for L in range(1, t)
                if abs(2 * L - t) <= 2]
        return np.median(np.stack(vals), axis=0)

    def solve(self, x, drop=0):
        """Delta(K) and the chiral bound-state energies at x, resumming with
        the highest orders less `drop`."""
        amps = self.resum(self.kin_series[:, : self.ng + 1 - drop], x, self.a, self.ng - drop)
        delta = (amps @ self.phases).reshape(self.nk, self.nk)
        uvals = self.resum(self.u_series[:, : self.n + 1 - drop], x, self.a, self.n - drop)
        U = np.zeros((len(self.support), len(self.support)))
        np.add.at(U, (self.rows, self.cols), uvals)
        Ub = self.basis.T @ U @ self.basis
        dmin = delta.min()

        def f(E):
            g = np.fft.ifft2(1.0 / (delta - E)).real
            G = g[self.diff[..., 0], self.diff[..., 1]]
            return np.linalg.det(np.eye(Ub.shape[0]) + self.basis.T @ G @ self.basis @ Ub)

        Es = dmin - np.geomspace(1e-7, 3.0, 300)[::-1]
        vals = [f(E) for E in Es]
        roots = []
        for (e1, f1), (e2, f2) in zip(zip(Es, vals), zip(Es[1:], vals[1:])):
            if np.sign(f1) != np.sign(f2):
                roots.append(brentq(f, e1, e2, xtol=1e-12))
        return dmin, roots


def solve_scan(s, xs):
    a = negative_singularity(s["gap0"][: len(s["gapK"])])
    tb = TwoBody(s, a)
    ex = exciton_energy(s)
    print("  two-body solution, Euler a = %.4f, median of near-diagonal Pade per amplitude" % a)
    print("     x     series E_ex   Delta(K)   E_ex      | one order less: Delta(K)   E_ex")
    for x in xs:
        cols = []
        for drop in (0, 1):
            dmin, roots = tb.solve(x, drop)
            cols.append("%8.4f  %8.4f%s" % (dmin, max(roots) if roots else float("nan"), "*" if len(roots) > 1 else " "))
        print("   %.3f  %8.4f    %s  |  %s" % (x, float(np.median(values(ex, a, x))), cols[0], cols[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--solve", action="store_true", help="also solve the resummed two-body problem against x")
    args = ap.parse_args()
    for path in args.json:
        s = load(path)
        n = s["n"]
        ex = exciton_energy(s)
        gap = s["gapK"][: n + 1]
        print("V/U = %s, s = %d, through x^%d" % (s["v"], s["nsites"], n))
        print("  E_ex       " + "  ".join(str(c) for c in ex))
        print("  E_ex       " + "  ".join("%.6g" % float(c) for c in ex))
        print("  Delta(K)   " + "  ".join("%.6g" % float(c) for c in gap))
        print("  binding    " + "  ".join("%.6g" % float(g - e) for g, e in zip(gap, ex)))
        a_ex = negative_singularity(ex)
        a_gap = negative_singularity(s["gap0"][: len(s["gapK"])])
        # E_ex crosses zero linearly, so plain Pade zeros locate it; Delta(K)
        # closes as a power, which dlog Pade captures and plain Pade overshoots.
        ze = zeros(ex, a_ex)
        zg = [float(z / (1 - z / a_gap)) for z, _ in cs.poles(cs.euler(s["gapK"], a_gap), 0.005, 0.95 * float(a_gap))]
        print("  unfrustrated-side singularities: E_ex x = -%.4f, Delta(0) x = -%.4f" % (a_ex, a_gap))
        print("  E_ex = 0 at      x_ex    = %s" % summary(ze))
        print("  Delta(K) = 0 at  x_Delta = %s" % summary(zg))
        if ze and zg:
            xg = float(np.median(zg))
            print("  E_ex at x_Delta     = %s" % summary(values(ex, a_ex, xg)))
        if args.solve:
            solve_scan(s, [0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11])
        print()


if __name__ == "__main__":
    main()
