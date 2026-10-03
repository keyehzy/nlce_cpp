#!/usr/bin/env python3
"""Table of kagome_dmrg.py results: per point the free run and the h = 0 end
of each pinning ramp, with the lowest-energy state marked.

Usage: kagome_dmrg_summary.py DIR
"""

import json
import sys
from pathlib import Path


def main():
    rows = []
    for f in sorted(Path(sys.argv[1]).glob("t*_ly*_chi*.json")):
        d = json.loads(f.read_text())
        states = {"free": d["free"]}
        for name in ("staggered", "uniform"):
            if name in d:
                states[name] = d[name][-1]
        rows.append((d["V"], d["Ly"], d["chi"], d["t"], states))
    print("%5s %3s %4s %6s | %-9s %12s %8s %8s %8s %7s | %s" % (
        "V/U", "Ly", "chi", "t/U", "state", "E/site", "k_up", "k_down", "xi_b", "P3", "dE vs lowest"))
    for v, ly, chi, t, states in sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3])):
        emin = min(m["E"] for m in states.values())
        for name, m in states.items():
            mark = "<-" if m["E"] - emin < 2e-6 else ""
            print("%5.3f %3d %4d %6.3f | %-9s %12.8f %+8.4f %+8.4f %8.2f %7.1e | %+.2e %s" % (
                v, ly, chi, t, name, m["E"], m["kappa_up"], m["kappa_down"], m["xi_b"], max(m["P_Nmax"]),
                m["E"] - emin, mark))
        print()


if __name__ == "__main__":
    main()
