#!/usr/bin/env python3
"""Convert nlce_run JSON output to the series/*.pkl format.

For each series_s{N}_v{tag}.json in SRC writes, into DST,

    res_site_tri_s{N}_v{tag}.pkl   EN, Hp, Hh        (series/nlce.py run)
    sq_site_tri_s{N}_v{tag}.pkl    S                 (series/nlce_sq.py run_sq)
    chi_site_tri_s{N}_v{tag}.pkl   chi               (series/nlce_chi.py run_chi)
    m0_site_tri_s{N}_v{tag}.pkl    m0                (series/nlce_chi.py run_m0)

with the same keys, values and dictionary order as the Python drivers.

Usage: to_pickles.py SRC [DST]
"""

import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path

# The Python drivers fill series with one shared Fraction(0); reusing a single
# object here makes pickle memoise it the same way, so the files match byte
# for byte and not only in value.
ZERO = Fraction(0)


def coefficient(x):
    f = Fraction(x)
    return ZERO if f == 0 else f


def convert(path, dst, quiet=False):
    d = json.loads(path.read_text())
    n, ng, nc = d["nsites"], d["order_gap"], d["order_chi"]
    v = Fraction(d["v"])
    tag = path.stem.split("_v", 1)[1]
    disp = [tuple(cd) for cd in d["displacements"]]
    series = lambda xs: [coefficient(x) for x in xs]

    res = {"order": ng, "nsites": n, "reliable_order": n - 1, "v": v, "lattice": "triangular",
           "EN": series(d["EN"]),
           "Hp": {cd: series(xs) for cd, xs in zip(disp, d["Hp"])},
           "Hh": {cd: series(xs) for cd, xs in zip(disp, d["Hh"])}}
    sq = {"order": ng, "nsites": n, "reliable_order": n - 1, "v": v,
          "S": {cd: series(xs) for cd, xs in zip(disp, d["S"]) if cd != (0, 0)}}
    chi = {"order": nc, "nsites": n, "reliable_order": n - 2, "v": v, "chi": series(d["chi"])}
    m0 = {"order": nc, "nsites": n, "reliable_order": n - 2, "v": v, "m0": series(d["m0"])}

    for prefix, obj in (("res", res), ("sq", sq), ("chi", chi), ("m0", m0)):
        out = dst / ("%s_site_tri_s%d_v%s.pkl" % (prefix, n, tag))
        with out.open("wb") as f:
            pickle.dump(obj, f, protocol=4)
        if not quiet:
            print(out)


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("series_s*_v*.json"))
    if not files:
        sys.exit("no series_s*_v*.json in %s" % src)
    for path in files:
        convert(path, dst)


if __name__ == "__main__":
    main()
