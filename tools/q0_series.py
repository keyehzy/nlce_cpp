#!/usr/bin/env python3
"""q = 0 series of an nlce_run lattice output in the usual -t convention.

From the series of one V/U (the JSON written by `nlce_run run`), per site:

    Delta(0) = sum_d (Hp[d] + Hh[d])           particle-hole gap at q = 0
    S(0)     = 3 + sum_{d != 0} S[d]           sum_j <b^dag_j b_0 + b_j b^dag_0>
    E/N                                        ground-state energy

summing over every displacement d, i.e. each canonical one times its orbit
size.  The on-site term of S is 2<n> + 1 = 3 at every order.  nlce_run's
hopping is +t sum b^dag_i b_j; the -t model follows by x -> -x, so a
coefficient of x^k carries an extra (-1)^k here.  On a bipartite lattice the
two signs are equivalent, and Delta(0) is the gap between the band edges.

Usage: q0_series.py SERIES.json [--float]
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path


def minus_t_sign(k):
    return 1 if k % 2 == 0 else -1


def lattice_q0(d):
    """Delta(q=0), S(q=0) and E/N, each a list of Fractions in the -t convention."""
    ng = d["order_gap"]
    gap = [Fraction(0)] * (ng + 1)
    sq = [Fraction(0)] * (ng + 1)
    sq[0] = Fraction(3)
    for disp, orbit, hp, hh, s in zip(d["displacements"], d["orbit_sizes"], d["Hp"], d["Hh"], d["S"]):
        for k in range(ng + 1):
            gap[k] += orbit * (Fraction(hp[k]) + Fraction(hh[k]))
            if disp != [0, 0]:
                sq[k] += orbit * Fraction(s[k])
    en = [Fraction(x) for x in d["EN"]]
    flip = lambda xs: [minus_t_sign(k) * x for k, x in enumerate(xs)]
    return {"gap": flip(gap), "S": flip(sq), "EN": flip(en)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--float", action="store_true", help="also print decimal values")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    q0 = lattice_q0(d)
    print("%s lattice, s = %d, V/U = %s, x = t/U, -t hopping" % (d.get("lattice", "triangular"), d["nsites"], d["v"]))
    for name, title in (("gap", "Delta(q=0)"), ("S", "S(q=0)"), ("EN", "E/N")):
        print(title)
        for k, a in enumerate(q0[name]):
            print("  %2d  %s%s" % (k, a, "    (%.15g)" % a if args.float and a.denominator != 1 else ""))


if __name__ == "__main__":
    main()
