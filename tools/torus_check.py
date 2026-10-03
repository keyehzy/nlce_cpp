#!/usr/bin/env python3
"""Check honeycomb or kagome lattice series against a periodic torus.

There is no published reference for these lattices, so this solves a torus,
the lattice modulo the superlattice spanned by S1 and S2, as a single finite
cluster with `nlce_run cluster` and compares its q = 0 series
(tools/q0_series.py) with `nlce_run run --lattice L`.  The torus bypasses the
cluster enumeration, symmetry reduction and lattice sums.

A process on the torus lifts to the infinite lattice unless its hops close a
loop around the torus, which takes at least L hops, L being the shortest
non-contractible cycle; the torus series are then exact through x^(L-1).
The default tori, the best with at most 16 sites, are a 14-site honeycomb
torus with L = 6, exact through x^5, the first order at which Delta sees a
hexagon, and a 15-site kagome torus with L = 4, exact through x^3, the first
order at which it sees a triangle.

Usage: torus_check.py --lattice honeycomb|kagome [--binary build/nlce_run]
                      [--nsites 7] [--s1 A,B --s2 A,B --loop L]
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

# Both are triangular lattices (a, b) without their hexagon centres, which are
# also the translations; default tori (s1, s2, shortest non-contractible cycle).
DIRS = [(1, 0), (0, 1), (1, -1)]
LATTICES = {
    "honeycomb": {"centre": lambda s: (s[0] - s[1]) % 3 == 0, "torus": ((-3, 9), (-7, 14), 6), "z": 3},
    "kagome": {"centre": lambda s: s[0] % 2 == 0 and s[1] % 2 == 0, "torus": ((2, 6), (0, 10), 4), "z": 4},
}


def torus_graph(lattice, s1, s2):
    """Sites and bonds of the lattice modulo the superlattice (s1, s2)."""
    centre = LATTICES[lattice]["centre"]
    is_site = lambda s: not centre(s)
    if not (centre(s1) and centre(s2)):
        raise ValueError("superlattice vectors must be %s translations" % lattice)
    det = s1[0] * s2[1] - s1[1] * s2[0]

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
    if 2 * len(bonds) != LATTICES[lattice]["z"] * n:
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
    ap.add_argument("--lattice", required=True, choices=sorted(LATTICES))
    ap.add_argument("--binary", default=str(HERE.parent / "build" / "nlce_run"))
    ap.add_argument("--nsites", type=int, default=7, help="lattice expansion size, at least loop")
    ap.add_argument("--s1")
    ap.add_argument("--s2")
    ap.add_argument("--loop", type=int, help="shortest non-contractible cycle of the torus")
    args = ap.parse_args()
    d1, d2, dloop = LATTICES[args.lattice]["torus"]
    s1 = tuple(int(x) for x in args.s1.split(",")) if args.s1 else d1
    s2 = tuple(int(x) for x in args.s2.split(",")) if args.s2 else d2
    order = (args.loop or dloop) - 1
    if args.nsites - 1 < order:
        ap.error("--nsites must be at least --loop")

    n, bonds = torus_graph(args.lattice, s1, s2)
    torus = torus_q0(args.binary, n, bonds, order)
    with tempfile.TemporaryDirectory(prefix="nlce_torus_") as tmp:
        subprocess.run([args.binary, "run", "--lattice", args.lattice, "--nsites", str(args.nsites), "--v", "0",
                        "--out", tmp], check=True, stderr=subprocess.DEVNULL)
        lattice = lattice_q0(json.loads(next(Path(tmp).glob("series_s*_v0.json")).read_text()))

    print("%s torus: %d sites, %d bonds, exact through x^%d" % (args.lattice, n, len(bonds), order))
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
