#!/usr/bin/env python3
"""Compare per-cluster series of nlce_run against the Python reference.

For every site-cluster class up to --smax sites (and every current pattern of
that class), runs `nlce_run cluster` and the exact-rational Python code in
series/pt.py, chi.py and neutral.py, and requires identical coefficients:

    E, Hp, Hh        lce.compute_graph   (gs_series, heff_series)
    corr             sq.compute_graph    (corr_series)
    chi, m0          chi.compute_key, neutral.compute_key

Usage: check_cluster.py [--smax 6] [--ng 7] [--nc 6] [--v 0,1/20,1/5]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE.parents[1] / "series"
BINARY = HERE.parent / "build" / "nlce_run"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smax", type=int, default=6)
    ap.add_argument("--ng", type=int, default=7)
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--v", default="0,1/20,1/5")
    ap.add_argument("--limit", type=int, default=0, help="max classes per size (0 = all)")
    args = ap.parse_args()

    os.environ["LCE_CACHE"] = tempfile.mkdtemp(prefix="nlce_check_")
    sys.path.insert(0, str(SERIES))
    import chi as chimod
    import lce
    import neutral
    import nlce
    import nlce_chi
    import sq
    from graphs import edges_from_cert

    geo = nlce.geometry_sites(args.smax, verbose=False)
    geo_chi = nlce_chi.geometry_chi_sites(args.smax, verbose=False)
    patterns = {}
    for (cert, pat) in geo_chi["mult"]:
        patterns.setdefault(cert, []).append(pat)

    by_size = {}
    for cert, s in geo["classes"].items():
        by_size.setdefault(s, []).append(cert)
    certs = []
    for s in sorted(by_size):
        group = sorted(by_size[s])
        certs += group[: args.limit] if args.limit else group

    checked = 0
    for vstr in args.v.split(","):
        v = Fraction(vstr)
        for cert in certs:
            nv, edges = edges_from_cert(cert)
            pats = patterns[cert]
            cmd = [str(BINARY), "cluster", "--nv", str(nv), "--ng", str(args.ng), "--nc", str(args.nc),
                   "--v", vstr, "--edges", ",".join("%d-%d" % e for e in edges) or "none"]
            for pat in pats:
                cmd += ["--pattern", "".join("+" if p > 0 else "-" for p in pat) or "none"]
            got = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
            F = lambda xs: [Fraction(x) for x in xs]

            _, gd = lce.compute_graph(cert, args.ng, v)
            want = {"E": [gd.E], "Hp": [gd.Hp[(u, w)] for u in range(nv) for w in range(nv)],
                    "Hh": [gd.Hh[(u, w)] for u in range(nv) for w in range(nv)]}
            _, corr = sq.compute_graph(cert, args.ng, v)
            zero = [Fraction(0)] * (args.ng + 1)
            want["corr"] = [corr[(u, w)] if u != w else zero for u in range(nv) for w in range(nv)]
            want["chi"] = [chimod.compute_key((cert, pat), args.nc, v)[1] for pat in pats]
            want["m0"] = [neutral.compute_key((cert, pat), args.nc, v)[1] for pat in pats]

            for name, rows in want.items():
                mine = [F(r) for r in got[name]]
                theirs = [list(r) for r in rows]
                if mine != theirs:
                    print("MISMATCH v=%s nv=%d edges=%s %s" % (vstr, nv, edges, name))
                    for i, (a, b) in enumerate(zip(mine, theirs)):
                        if a != b:
                            print("  row %d\n   cpp %s\n   py  %s" % (i, a, b))
                            break
                    sys.exit(1)
            checked += 1
        print("v=%s: %d classes identical" % (vstr, len(certs)), flush=True)
    print("all %d cluster checks passed" % checked)


if __name__ == "__main__":
    main()
