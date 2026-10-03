#!/usr/bin/env python3
"""Compare lattice series with Elstner and Monien, cond-mat/9905367.

Tables I-III of the paper list the strong-coupling series of the single
particle-hole gap Delta(q=0) and the equal-time structure factor

    S(q=0) = sum_j <b^dag_j b_0 + b_j b^dag_0>

of the Bose-Hubbard model at unit filling on the square lattice, the
triangular lattice and the chain, through x^13 with x = t/U; Appendix A gives
the ground-state energy of the chain through x^6.  The paper's hopping is
-t sum b^dag_i b_j, as in tools/q0_series.py, which assembles these sums
from the nlce_run series at V/U = 0.

Some high-order entries of the paper are printed with a decimal point, so
they were rounded; those are compared to a relative 1e-12 and reported as
approximate, as are the few rounded entries printed as fractions
(ROUNDED_FRACTIONS), to their printed denominator.  Known misprints (ERRATA)
are compared in corrected form.  Every other entry must match exactly.

Runs `nlce_run run --lattice L --nsites N --v 0`, or reads an existing JSON
file with --json.  Exits 1 on any mismatch.

Usage: paper_check.py --lattice chain --nsites 14 [--binary build/nlce_run]
       paper_check.py --json out/series_s14_v0.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q0_series import lattice_q0  # noqa: E402

# Coefficients a_0 .. a_13 as printed in the paper.
PAPER = {
    "square": {
        "gap": [
            "1", "-12", "-22", "-264", "-15659 / 10", "-656984 / 25", "-513092341 / 2250",
            "-13396365654 / 3375", "-2194497431888101 / 56700000", "-523244582353596437 / 744187500",
            "-4749112579154967367231 / 625117500000", "-6676218845916748474723399 / 49228003125000",
            "-5669114326328304841982042447 / 3508972062750000",
            "-3473317126780784521271398100 / 126449317253259",
        ],
        "S": [
            "3", "32", "432", "6656", "99632", "14154496 / 9", "663550400 / 27", "31905307840 / 81",
            "7618958766796 / 1215", "12934227681606432 / 127575", "14570617373829713351 / 8930250",
            "19905912307529372064253 / 750141000",
            "17231660529006598072716626038 / 40068302174901",
            "47514032474492554578981737799.4028 / 6764778289269",
        ],
    },
    "triangular": {
        "gap": [
            "1", "-18", "-81", "-819", "-54891 / 4", "-11459377 / 50", "-6764830501 / 1500",
            "-3957593443549 / 45000", "-23724030434424597 / 12600000", "-79054659543137812691 / 1984500000",
            "-49507676563116513700717 / 55566000000", "-19752788544107.0312336222045518956",
            "-454652221307484.008486685913608139", "-10398395856680122.5582305335816799",
        ],
        "S": [
            "3", "48", "1080", "25344", "613016", "15108128", "3391856144 / 9", "28444402112 / 3",
            "97228564590772 / 405", "86595555599744452 / 14175", "2787620342817465447617 / 17860500",
            "82527451969123616224435919 / 20628877500", "102821325219551430.224270929006395",
            "2648736908451106507.61278132531748",
        ],
    },
    "chain": {
        "gap": [
            "1", "-6", "5", "6", "287 / 20", "17463 / 150", "-1806729 / 3000", "73674531 / 22500",
            "-297690613629 / 16200000", "14666046468323 / 121500000", "-6295148943458549 / 7290000000",
            "114441271150219589 / 18225000000", "-422231271662550684871 / 9185400000000",
            "1473292023890353319230511 / 4340101500000000",
        ],
        "S": [
            "3", "16", "72", "320", "4120 / 3", "48544 / 9", "596992 / 27", "2396512 / 27", "27653840 / 81",
            "1329678", "22598877209 / 4375", "1275342277201 / 65610", "4055776421430107 / 55112400",
            "203597243119484303 / 723350250",
        ],
        # Appendix A: E/N = -4x^2 + 4x^4 + 272/9 x^6.
        "EN": ["0", "0", "-4", "0", "4", "0", "272 / 9"],
    },
}

# Misprints in the paper, (lattice, series, order) -> corrected entry.  The
# chain's S at x^10 is printed over 4375; brute-force Rayleigh-Schroedinger
# perturbation theory in the full Hilbert space of a 12-site ring, exact
# through x^10, gives the same numerator over 4374 = 2 * 3^7.
ERRATA = {
    ("chain", "S", 10): "22598877209 / 4374",
}

# Entries printed as p / q that are not exact: the paper's coefficient was
# evaluated in floating point and p rounded to an integer over a chosen q, as
# the decimal numerators of other entries show.  These must satisfy
# p = round(a * q) for our exact a.  The square lattice's gap at x^13, with
# no factor 2 or 5 in q unlike every other gap entry, is likely another.
ROUNDED_FRACTIONS = {
    ("square", "gap", 12),
    ("square", "S", 12),
}

# Relative tolerance for the entries printed with a decimal point.
REL_TOL = Decimal("1e-12")


def parse(entry):
    """(value, exact) for an entry 'p', 'p / q' or a decimal form."""
    parts = [p.strip() for p in entry.split("/")]
    exact = not any("." in p for p in parts)
    if exact:
        value = Fraction(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1)
    else:
        value = Decimal(parts[0]) / (Decimal(parts[1]) if len(parts) > 1 else 1)
    return value, exact


def compare(lattice, ours):
    """Prints the comparison; returns (exact matches, approximate matches, failures)."""
    exact_ok = approx_ok = bad = 0
    for name, ref in PAPER[lattice].items():
        series = ours[name]
        n = min(len(series), len(ref))
        print("%s %s through x^%d" % (lattice, name, n - 1))
        for k in range(n):
            erratum = ERRATA.get((lattice, name, k))
            value, exact = parse(erratum or ref[k])
            got = series[k]
            if (lattice, name, k) in ROUNDED_FRACTIONS:
                ok = round(got * value.denominator) == value.numerator
                exact = False
                tag = "matches paper's rounding to /%d" % value.denominator if ok else "DIFFERENT"
            elif exact:
                ok = got == value
                tag = ("exact (paper misprints %s)" % ref[k] if erratum else "exact") if ok else "DIFFERENT"
            else:
                got_dec = Decimal(got.numerator) / Decimal(got.denominator)
                ok = abs(got_dec - value) <= REL_TOL * abs(value)
                tag = "approx (paper rounded)" if ok else "DIFFERENT"
            print("  %2d  %-50s %s" % (k, str(got), tag))
            if not ok:
                print("      paper: %s" % ref[k])
                bad += 1
            elif exact:
                exact_ok += 1
            else:
                approx_ok += 1
    return exact_ok, approx_ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", choices=sorted(PAPER))
    ap.add_argument("--nsites", type=int)
    ap.add_argument("--binary", default=str(HERE.parent / "build" / "nlce_run"))
    ap.add_argument("--threads", type=int)
    ap.add_argument("--json", help="existing nlce_run output at V/U = 0 instead of a fresh run")
    args = ap.parse_args()

    if args.json:
        d = json.loads(Path(args.json).read_text())
    else:
        if not args.lattice or not args.nsites:
            ap.error("--lattice and --nsites are required without --json")
        with tempfile.TemporaryDirectory(prefix="nlce_paper_") as tmp:
            cmd = [args.binary, "run", "--lattice", args.lattice, "--nsites", str(args.nsites), "--v", "0",
                   "--out", tmp]
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
            d = json.loads(next(Path(tmp).glob("series_s*_v0.json")).read_text())
    lattice = d["lattice"]
    if d["v"] != "0":
        sys.exit("the paper's series are for V/U = 0, not %s" % d["v"])
    exact_ok, approx_ok, bad = compare(lattice, lattice_q0(d))
    print("%s s=%d: %d exact, %d approximate, %d different" % (lattice, d["nsites"], exact_ok, approx_ok, bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
