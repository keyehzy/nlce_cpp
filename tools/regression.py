#!/usr/bin/env python3
"""Fast end-to-end regression against the reference series pickles.

An s-site expansion is exact through x^(s-1) (x^(s-2) for chi and m0), so a
small run must reproduce the leading coefficients of every larger reference
series.  Runs `nlce_run run --nsites N` for each V/U present in REF, converts
the output with to_pickles.py, and compares every series against the
reference of the smallest size above N.  Displacements absent from the small
run must have vanishing leading coefficients in the reference.

Exits 77 (reported as skipped by ctest) when REF holds no larger reference.

Usage: regression.py [--binary build/nlce_run] [--nsites 7] [--ref validate]
"""

import argparse
import pickle
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import to_pickles  # noqa: E402

SKIP = 77
NAME = re.compile(r"^(res|sq|chi|m0)_site_tri_s(\d+)_v(\w+)\.pkl$")


def tag_to_rational(tag):
    return tag.replace("over", "/")


def compare(new, ref):
    """Names of the entries of `new` that are not prefixes of `ref`."""
    bad = []
    if new["v"] != ref["v"]:
        bad.append("v")
    for key, ours in new.items():
        if isinstance(ours, list):
            if ours != ref[key][: len(ours)]:
                bad.append(key)
        elif isinstance(ours, dict):
            n = len(next(iter(ours.values())))
            zero = [Fraction(0)] * n
            theirs = ref[key]
            for cd in sorted(set(ours) | set(theirs)):
                if cd not in theirs or ours.get(cd, zero) != theirs[cd][:n]:
                    bad.append("%s%s" % (key, cd))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(HERE.parent / "build" / "nlce_run"))
    ap.add_argument("--nsites", type=int, default=7)
    ap.add_argument("--ref", default=str(HERE.parent / "validate"))
    args = ap.parse_args()

    refs = {}
    for path in Path(args.ref).glob("*.pkl"):
        m = NAME.match(path.name)
        if m and int(m.group(2)) > args.nsites:
            refs.setdefault(int(m.group(2)), {})[(m.group(1), m.group(3))] = path
    if not refs:
        print("no reference series above s=%d in %s; skipping" % (args.nsites, args.ref))
        sys.exit(SKIP)
    size = min(refs)
    refs = refs[size]
    tags = sorted({tag for _, tag in refs})

    with tempfile.TemporaryDirectory(prefix="nlce_regression_") as tmp:
        tmp = Path(tmp)
        subprocess.run([args.binary, "run", "--nsites", str(args.nsites), "--out", str(tmp),
                        "--v", ",".join(tag_to_rational(t) for t in tags)],
                       check=True, stderr=subprocess.DEVNULL)
        for path in sorted(tmp.glob("series_s*_v*.json")):
            to_pickles.convert(path, tmp, quiet=True)

        failures = 0
        for (prefix, tag), ref_path in sorted(refs.items()):
            new_path = tmp / ("%s_site_tri_s%d_v%s.pkl" % (prefix, args.nsites, tag))
            bad = compare(pickle.loads(new_path.read_bytes()), pickle.loads(ref_path.read_bytes()))
            status = "ok" if not bad else "DIFFERENT: " + ", ".join(bad[:5])
            print("%-28s vs s=%d  %s" % (new_path.name, size, status))
            failures += bool(bad)
    print("%d of %d series differ" % (failures, len(refs)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
