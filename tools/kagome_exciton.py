#!/usr/bin/env python3
"""Doublon-holon excitons on the kagome lattice, as exact series.

The kagome counterpart of tools/exciton_series.py.  Reads `nlce_run run
--lattice kagome --dh 1` output: the pair amplitudes Hp_pairs, Hh_pairs
(through x^(N-1)) and the doublon-holon interaction Idh (through x^(N-2)),
whose keys fix the holon's sublattice h and the relative coordinate
r = doublon - holon before and after.  At total momentum zero the
two-particle effective Hamiltonian on the states (h, r) is

    H((h', r') <- (h, r)) = Hp(doublon hop) + Hh(holon hop) + Idh.

At x = 0 the twelve adjacent states per cell (six bonds, two orientations)
have energy 1 - V and every other pair 1.  They carry the regular
representation of D6, so each one-dimensional representation occurs once
among them, and its state has an ordinary Rayleigh-Schroedinger series
for V > 0.  The channels computed, each named by its amplitude on the
adjacent pair (doublon i, holon j):

    staggered   sign(j -> i) of the staggered current: the state K|Mott>
                of the sqrt3 x sqrt3 chiral current (nlce_run --current)
    uniform     the same for the uniform (q = 0) chiral current
    trimer      +1 on up triangles, -1 on down ones, symmetric in i, j:
                the breathing (trimerised) bond order, not chiral

A chiral Mott insulator appears where a chiral exciton energy reaches zero
while the particle-hole gap (tools/kagome_cmi.py) is still open.  Printed per
channel: the series, its zero (near-diagonal Pade after an Euler transform
that removes its nearest singularity on the unfrustrated side, as in
exciton_series.py), and its value where the gap closes.

--solve instead resums every amplitude (Hp, Hh between two sites, Idh per
key) after the Euler transform of Delta(0), whose radius is that of the
vacuum, about twice the exciton series', and diagonalises the two-body
problem at total momentum zero in a box of relative coordinates, projected
on each channel's representation.  The lowest level of the channel is a
bound exciton where it lies below the gap; above it the levels are the
box's discretised continuum.

Usage: kagome_exciton.py SERIES.json [SERIES.json ...] [--channels staggered,uniform,trimer]
       kagome_exciton.py SERIES.json --solve [--radius 10] [--xs 0.05,0.1,...]
"""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

import cmi_series as cs
import exciton_series as es
import kagome_cmi as kc

DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def is_site(s):
    return not (s[0] % 2 == 0 and s[1] % 2 == 0)


def anchor(s):
    return kc.ANCHORS[kc.residue(s)]


def add(s, t):
    return (s[0] + t[0], s[1] + t[1])


def sub(s, t):
    return (s[0] - t[0], s[1] - t[1])


def staggered_sign(p, q):
    """lattice.cpp current_sign(p, q)."""
    lp, lq = (2 * p[0] + p[1]) % 3, (2 * q[0] + q[1]) % 3
    return 1 if (lq - lp) % 3 == 1 else -1


def up_triangle(p, q):
    """Whether the kagome triangle of the bond {p, q} points up (lattice.cpp)."""
    u, d = p, sub(q, p)
    if d not in ((1, 0), (0, 1), (1, -1)):
        u, d = q, sub(p, q)
    third = {(1, 0): (0, 1), (0, 1): (1, 0), (1, -1): (0, -1)}[d]
    return is_site(add(u, third))


CHANNELS = {
    "staggered": lambda j, i: staggered_sign(j, i),
    "uniform": lambda j, i: -staggered_sign(j, i) if up_triangle(j, i) else staggered_sign(j, i),
    "trimer": lambda j, i: 1 if up_triangle(j, i) else -1,
}


def group_images(key):
    """Point-group images of an Idh key (h', r', h, r), holons normalised."""
    h2, r2, h, r = key
    seen = {key}
    frontier = [key]
    while frontier:
        nxt = []
        for a2, s2, a, s in frontier:
            for g in (cs.ROTATE, cs.MIRROR):
                im = (anchor(g(a2)), g(s2), anchor(g(a)), g(s))
                if im not in seen:
                    seen.add(im)
                    nxt.append(im)
        frontier = nxt
    return seen


def load(path):
    d = json.loads(Path(path).read_text())
    if "Idh" not in d or "dh_holons" not in d:
        raise SystemExit("%s: no kagome Idh series; run nlce_run --lattice kagome --dh 1" % path)
    n = d["order_dh"]
    ser = lambda xs: [Fraction(x) for x in xs[: n + 1]]
    # hops[which][source sublattice] = [(target - source, series)]
    hops = {"p": {r: [] for r in kc.ANCHORS}, "h": {r: [] for r in kc.ANCHORS}}
    hops_full = {"p": {r: [] for r in kc.ANCHORS}, "h": {r: [] for r in kc.ANCHORS}}
    for key, hp, hh in zip(d["pairs"], d["Hp_pairs"], d["Hh_pairs"]):
        p, q = (key[0], key[1]), (key[2], key[3])
        sp, sh = ser(hp), ser(hh)
        fp, fh = [Fraction(c) for c in hp], [Fraction(c) for c in hh]
        for x, y in kc.pair_images(p, q):
            # amplitude from site y to site x
            hops["p"][kc.residue(y)].append((sub(x, y), sp))
            hops["h"][kc.residue(y)].append((sub(x, y), sh))
            hops_full["p"][kc.residue(y)].append((sub(x, y), fp))
            hops_full["h"][kc.residue(y)].append((sub(x, y), fh))
    idh = {}
    for (r2a, r2b, ra, rb), (h2a, h2b, ha, hb), xs in zip(d["dh_keys"], d["dh_holons"], d["Idh"]):
        xs = ser(xs)
        if not any(xs):
            continue
        for im in group_images(((h2a, h2b), (r2a, r2b), (ha, hb), (ra, rb))):
            idh[im] = xs
    base = kc.load(path)
    return {"v": Fraction(d["v"]), "nsites": d["nsites"], "n": n, "hops": hops, "hops_full": hops_full, "idh": idh,
            "base": base}


def hop_distance(r):
    a, b = r
    return max(abs(a), abs(b), abs(a + b))


def hamiltonian(s, radius):
    """H[((h', r'), (h, r))] for 0 < |r|, |r'| <= radius."""
    n = s["n"]
    states = [(h, r) for h in kc.ANCHORS.values() for a in range(-radius, radius + 1)
              for b in range(-radius, radius + 1)
              for r in [(a, b)] if 0 < hop_distance(r) <= radius and is_site(add(h, r))]
    index = set(states)
    H = {}

    def put(key, xs):
        cur = H.setdefault(key, [Fraction(0)] * (n + 1))
        for k in range(n + 1):
            cur[k] += xs[k]

    for h, r in states:
        i = add(h, r)
        for e, xs in s["hops"]["p"][kc.residue(i)]:
            r2 = sub(add(i, e), h)
            if (h, r2) in index:
                put(((h, r2), (h, r)), xs)
        for e, xs in s["hops"]["h"][kc.residue(h)]:
            j2 = add(h, e)
            h2 = anchor(j2)
            r2 = sub(i, j2)
            if (h2, r2) in index:
                put(((h2, r2), (h, r)), xs)
    for (h2, r2, h, r), xs in s["idh"].items():
        if (h2, r2) in index and (h, r) in index:
            put(((h2, r2), (h, r)), xs)
    return states, H


def exciton_energy(s, channel):
    """Rayleigh-Schroedinger series of the exciton energy in one channel."""
    n = s["n"]
    v = s["v"]
    if v <= 0:
        raise SystemExit("the excitons are degenerate with the continuum at V = 0")
    states, H = hamiltonian(s, n + 2)
    diag0 = {st: H.get((st, st), [Fraction(0)])[0] for st in states}
    for (a, b), xs in H.items():
        if a != b and xs[0] != 0:
            raise SystemExit("off-diagonal amplitude at x^0")
    f = CHANNELS[channel]
    shell = {}
    for h in kc.ANCHORS.values():
        for d in DIRS:
            if is_site(add(h, d)):
                shell[(h, d)] = Fraction(f(h, add(h, d)))
    assert len(shell) == 12
    norm = sum(c * c for c in shell.values())
    e0 = 1 - v
    assert all(diag0[st] == e0 for st in shell)
    cols = [dict() for _ in range(n + 1)]
    for (a, b), xs in H.items():
        for m in range(1, n + 1):
            if xs[m]:
                cols[m].setdefault(b, []).append((a, xs[m]))

    def apply(m, vec):
        out = {}
        for b, c in vec.items():
            for a, w in cols[m].get(b, ()):
                out[a] = out.get(a, 0) + w * c
        return out

    psi = [dict(shell)]
    energy = [e0]
    for k in range(1, n + 1):
        ek = Fraction(0)
        for m in range(1, k + 1):
            w = apply(m, psi[k - m])
            ek += sum(w.get(st, 0) * c for st, c in shell.items()) / norm
        energy.append(ek)
        if k == n:
            break
        rhs = {}
        for m in range(1, k + 1):
            for st, c in apply(m, psi[k - m]).items():
                rhs[st] = rhs.get(st, 0) + c
            for st, c in psi[k - m].items():
                rhs[st] = rhs.get(st, 0) - energy[m] * c
        nxt = {}
        for st, c in rhs.items():
            if st in shell:
                continue
            if c:
                nxt[st] = c / (e0 - diag0[st])
        along = sum(rhs.get(st, 0) * c for st, c in shell.items()) / norm
        rest = {st: rhs.get(st, 0) - along * c for st, c in shell.items()}
        if along or any(rest.values()):
            raise SystemExit("%s sector not closed at order %d" % (channel, k))
        psi.append(nxt)
    return energy


def gap_of(s, n):
    """The particle-hole gap series E_p + E_h at the band minima, through x^n."""
    base = s["base"]
    gap, _, _ = kc.gap_series(base, "Gamma", "Gamma")
    return gap[: n + 1], gap


def group_elements():
    """The twelve D6 elements as functions of a site, identity first."""
    ops = [lambda t: t]
    for _ in range(5):
        prev = ops[-1]
        ops.append(lambda t, prev=prev: cs.ROTATE(prev(t)))
    return ops + [lambda t, g=g: cs.MIRROR(g(t)) for g in ops]


class TwoBody:
    """The two-body problem at total momentum zero with resummed amplitudes."""

    def __init__(self, s, radius):
        self.s = s
        n = s["n"]
        ng = s["base"]["ng"]
        self.states = [(h, r) for h in kc.ANCHORS.values() for a in range(-radius, radius + 1)
                       for b in range(-radius, radius + 1)
                       for r in [(a, b)] if 0 < hop_distance(r) <= radius and is_site(add(h, r))]
        index = {st: i for i, st in enumerate(self.states)}
        # Atoms: every distinct amplitude series, padded to the gap order.
        atoms, rows, cols, which = [], [], [], []
        atom_of = {}

        def atom(xs):
            key = id(xs)
            if key not in atom_of:
                atom_of[key] = len(atoms)
                atoms.append([float(c) for c in xs] + [0.0] * (ng + 1 - len(xs)))
            return atom_of[key]

        self.hop_len = ng + 1
        for st, col in index.items():
            h, r = st
            i = add(h, r)
            for e, xs in s["hops_full"]["p"][kc.residue(i)]:
                r2 = sub(add(i, e), h)
                if (h, r2) in index:
                    rows.append(index[(h, r2)]); cols.append(col); which.append(atom(xs))
            for e, xs in s["hops_full"]["h"][kc.residue(h)]:
                j2 = add(h, e)
                st2 = (anchor(j2), sub(i, j2))
                if st2 in index:
                    rows.append(index[st2]); cols.append(col); which.append(atom(xs))
        self.n_hop_atoms = len(atoms)
        for (h2, r2, h, r), xs in s["idh"].items():
            if (h2, r2) in index and (h, r) in index:
                rows.append(index[(h2, r2)]); cols.append(index[(h, r)]); which.append(atom(xs))
        self.atoms = np.array(atoms)
        self.rows, self.cols, self.which = np.array(rows), np.array(cols), np.array(which)
        self.n, self.ng = n, ng
        self.a = float(es.negative_singularity(s["base"]["gap0"]))
        # Group action on the box.
        perms = []
        for g in group_elements():
            perm = np.empty(len(self.states), dtype=int)
            for k, (h, r) in enumerate(self.states):
                perm[k] = index[(anchor(g(h)), g(r))]
            perms.append(perm)
        self.perms = perms

    def resum(self, x, drop=0):
        """Every atom at x: the median of the near-diagonal Pade of its Euler
        transform, the gap-order atoms (Hp, Hh) and the Idh atoms each with
        their own number of terms, less `drop`."""
        out = np.empty(len(self.atoms))
        for lo, hi, order in ((0, self.n_hop_atoms, self.ng), (self.n_hop_atoms, len(self.atoms), self.n)):
            if hi == lo:
                continue
            t = order - drop
            series = self.atoms[lo:hi, : t + 1] @ es.euler_matrix(t, self.a).T
            z = x / (1 + x / self.a)
            vals = [es.pade_values(series[:, : tt + 1], L, tt - L, z) for tt in (t, t - 1) for L in range(1, tt)
                    if abs(2 * L - tt) <= 2]
            out[lo:hi] = np.median(np.stack(vals), axis=0)
        return out

    def channel_basis(self, channel):
        f = CHANNELS[channel]
        vec = np.zeros(len(self.states))
        for k, (h, r) in enumerate(self.states):
            if hop_distance(r) == 1:
                vec[k] = f(h, add(h, r))
        chars = [float(vec[p] @ vec) / float(vec @ vec) for p in self.perms]
        P = np.zeros((len(self.states), len(self.states)))
        for c, p in zip(chars, self.perms):
            P[p, np.arange(len(self.states))] += c / 12
        w, v = np.linalg.eigh((P + P.T) / 2)
        return v[:, w > 0.5]

    def levels(self, x, basis, drop=0, count=3):
        vals = self.resum(x, drop)
        H = np.zeros((len(self.states), len(self.states)))
        np.add.at(H, (self.rows, self.cols), vals[self.which])
        Hc = basis.T @ H @ basis
        ev, evec = np.linalg.eig(Hc)
        order = np.argsort(ev.real)
        out = []
        for k in order[:count]:
            psi = basis @ evec[:, k].real
            w = psi ** 2 / (psi @ psi)
            near = sum(wk for wk, (h, r) in zip(w, self.states) if hop_distance(r) <= 2)
            out.append((ev[k].real, abs(ev[k].imag), near))
        return out


def solve(s, xs, radius, channels):
    tb = TwoBody(s, radius)
    full_gap = gap_of(s, s["base"]["ng"])[1]
    a_gap = es.negative_singularity(s["base"]["gap0"])
    print("  two-body solution, box radius %d (%d states), Euler a = %.4f" % (radius, len(tb.states), tb.a))
    print("  levels: energy (weight within hop distance 2); '*' complex pair")
    for ch in channels:
        basis = tb.channel_basis(ch)
        print("  %s channel (%d states)" % (ch, basis.shape[1]))
        print("       x    gap (series)   lowest levels                                  | one order less: lowest")
        for x in xs:
            g = np.median(es.values(full_gap, a_gap, x))
            lv = tb.levels(x, basis)
            lv1 = tb.levels(x, basis, drop=1, count=1)
            fmt = lambda e, im, w: "%8.4f%s(%.2f)" % (e, "*" if im > 1e-9 else " ", w)
            print("   %.3f  %8.4f       %s  |  %s" % (x, g, "  ".join(fmt(*l) for l in lv), fmt(*lv1[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--channels", default="staggered,uniform,trimer")
    ap.add_argument("--solve", action="store_true", help="solve the resummed two-body problem against x")
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--xs", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4")
    args = ap.parse_args()
    for path in args.json:
        s = load(path)
        n = s["n"]
        if args.solve:
            print("V/U = %s, s = %d, Idh through x^%d" % (s["v"], s["nsites"], n))
            solve(s, [float(x) for x in args.xs.split(",")], args.radius, args.channels.split(","))
            print()
            continue
        gap_n, gap = gap_of(s, n)
        a_gap = es.negative_singularity(s["base"]["gap0"])
        zg = [float(z / (1 - z / a_gap)) for z, _ in cs.poles(cs.euler(gap, a_gap), 0.005, 0.95 * float(a_gap))]
        pz = es.zeros(gap, a_gap, hi=2.0)
        print("V/U = %s, s = %d, excitons through x^%d, gap through x^%d" % (s["v"], s["nsites"], n, len(gap) - 1))
        print("  gap        " + "  ".join("%.6g" % float(c) for c in gap))
        print("  unfrustrated-side singularity of Delta(0): x = -%.4f" % a_gap)
        print("  gap: dlog Pade poles x_Delta = %s; Pade zeros %s" % (es.summary(zg), es.summary(pz)))
        for ch in args.channels.split(","):
            ex = exciton_energy(s, ch)
            print("  %-9s  E_ex  " % ch + "  ".join(str(c) for c in ex))
            print("  %-9s  E_ex  " % "" + "  ".join("%.6g" % float(c) for c in ex))
            print("  %-9s  bind  " % "" + "  ".join("%.6g" % float(g - e) for g, e in zip(gap_n, ex)))
            a_ex = es.negative_singularity(ex)
            ze = es.zeros(ex, a_ex, hi=2.0)
            print("  %-9s  unfrustrated-side singularity x = -%.4f; E_ex = 0 at x_ex = %s" % ("", a_ex, es.summary(ze)))
            for x in (0.05, 0.1, 0.15, 0.2):
                ve = es.values(ex, a_ex, x)
                vg = es.values(gap, a_gap, x)
                print("  %-9s  x = %.2f  E_ex = %s   gap = %s" % ("", x, es.summary(ve), es.summary(vg)))
        print()


if __name__ == "__main__":
    main()
