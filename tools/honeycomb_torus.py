#!/usr/bin/env python3
"""Check the honeycomb lattice series against a periodic honeycomb torus.

There is no published reference for the honeycomb lattice, so this solves a
torus, the honeycomb modulo the superlattice spanned by S1 and S2, as a
single finite cluster with `nlce_run cluster` and compares its q = 0 series
(tools/q0_series.py) with `nlce_run run --lattice honeycomb`.  The torus
bypasses the cluster enumeration, symmetry reduction and lattice sums.

A process on the torus lifts to the infinite lattice unless its hops close a
loop around the torus, which takes at least L hops, L being the shortest
non-contractible cycle; the torus series are then exact through x^(L-1).
The default 14-site torus has L = 6, so Delta(0), S(0) and E/N must agree
through x^5, the first order at which Delta sees a hexagon.

Usage: honeycomb_torus.py [--binary build/nlce_run] [--nsites 7]
                          [--s1 -3,9 --s2 -7,14 --loop 6]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q0_series import lattice_q0, minus_t_sign  # noqa: E402

# Honeycomb sites are the triangular sites (a, b) with (2a + b) mod 3 != 0.
DIRS = [(1, 0), (0, 1), (1, -1)]


def is_site(s):
    return (2 * s[0] + s[1]) % 3 != 0


def torus_graph(s1, s2):
    """Sites and bonds of the honeycomb modulo the superlattice (s1, s2)."""
    det = s1[0] * s2[1] - s1[1] * s2[0]
    if det % 3:
        raise ValueError("superlattice vectors must be honeycomb translations")

    def cell(s):
        # coordinates in the (s1, s2) basis modulo 1
        u = Fraction(s[0] * s2[1] - s[1] * s2[0], det)
        v = Fraction(s1[0] * s[1] - s1[1] * s[0], det)
        return (u - (u.numerator // u.denominator), v - (v.numerator // v.denominator))

    reach = 2 * (abs(s1[0]) + abs(s1[1]) + abs(s2[0]) + abs(s2[1]))
    rep = {}
    for a in range(-reach, reach + 1):
        for b in range(-reach, reach + 1):
            if is_site((a, b)):
                rep.setdefault(cell((a, b)), (a, b))
    index = {c: i for i, c in enumerate(sorted(rep))}
    bonds = set()
    for c, s in rep.items():
        for d in DIRS:
            t = (s[0] + d[0], s[1] + d[1])
            if is_site(t):
                i, j = index[c], index[cell(t)]
                if i == j:
                    raise ValueError("torus too small: a bond closes on itself")
                bonds.add((min(i, j), max(i, j)))
    n = len(index)
    if len(bonds) != 3 * n // 2:
        raise ValueError("torus too small: repeated bonds")
    return n, sorted(bonds)


def torus_q0(binary, n, bonds, order):
    """Delta(0), S(0) and E/N of the torus in the -t convention."""
    out = subprocess.run([binary, "cluster", "--nv", str(n), "--edges", ",".join("%d-%d" % b for b in bonds),
                          "--ng", str(order), "--nc", "0", "--v", "0"],
                         check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    gap = [Fraction(0)] * (order + 1)
    sq = [Fraction(0)] * (order + 1)
    for u in range(n):
        for w in range(n):
            q = u * n + w
            for k in range(order + 1):
                gap[k] += Fraction(d["Hp"][q][k]) + Fraction(d["Hh"][q][k])
                if u != w:
                    sq[k] += Fraction(d["corr"][q][k])
    en = [Fraction(x) for x in d["E"][0]]
    per_site = lambda xs: [minus_t_sign(k) * x / n for k, x in enumerate(xs)]
    sq = per_site(sq)
    sq[0] += 3  # on-site term 2<n> + 1
    return {"gap": per_site(gap), "S": sq, "EN": per_site(en)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(HERE.parent / "build" / "nlce_run"))
    ap.add_argument("--nsites", type=int, default=7, help="lattice expansion size, at least loop + 1")
    ap.add_argument("--s1", default="-3,9")
    ap.add_argument("--s2", default="-7,14")
    ap.add_argument("--loop", type=int, default=6, help="shortest non-contractible cycle of the torus")
    args = ap.parse_args()
    s1 = tuple(int(x) for x in args.s1.split(","))
    s2 = tuple(int(x) for x in args.s2.split(","))
    order = args.loop - 1
    if args.nsites - 1 < order:
        ap.error("--nsites must be at least --loop")

    n, bonds = torus_graph(s1, s2)
    torus = torus_q0(args.binary, n, bonds, order)
    with tempfile.TemporaryDirectory(prefix="nlce_honeycomb_") as tmp:
        subprocess.run([args.binary, "run", "--lattice", "honeycomb", "--nsites", str(args.nsites), "--v", "0",
                        "--out", tmp], check=True, stderr=subprocess.DEVNULL)
        lattice = lattice_q0(json.loads(next(Path(tmp).glob("series_s*_v0.json")).read_text()))

    print("honeycomb torus: %d sites, %d bonds, exact through x^%d" % (n, len(bonds), order))
    bad = 0
    for name in ("gap", "S", "EN"):
        same = torus[name] == lattice[name][: order + 1]
        bad += not same
        print("  %-3s %s  %s" % (name, "agree" if same else "DIFFERENT", " ".join(str(x) for x in torus[name])))
        if not same:
            print("      lattice  %s" % " ".join(str(x) for x in lattice[name][: order + 1]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
