#!/usr/bin/env python3
"""Check that regenerated series pickles equal a reference set exactly.

Compares every *_site_tri_s{N}_v*.pkl present in NEW against the file of the
same name in REF (default: validate/, the reference s = 9 and s = 10 series),
as Python objects (Fraction values, dictionary contents).  Also reports
whether the files are byte-identical; with --bytes a file that is not counts
as a difference.

Usage: compare_pickles.py [--bytes] NEW [REF]
"""

import argparse
import hashlib
import pickle
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", action="store_true", help="also require byte-identical files")
    ap.add_argument("new")
    ap.add_argument("ref", nargs="?", default=str(Path(__file__).resolve().parents[1] / "validate"))
    args = ap.parse_args()
    new, ref = Path(args.new), Path(args.ref)
    files = sorted(new.glob("*_site_tri_s*_v*.pkl"))
    if not files:
        sys.exit("no pickles in %s" % new)
    bad = 0
    for path in files:
        other = ref / path.name
        if not other.exists():
            print("%-34s  no reference" % path.name)
            continue
        a = pickle.loads(path.read_bytes())
        b = pickle.loads(other.read_bytes())
        same_bytes = hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(other.read_bytes()).digest()
        if a == b and (same_bytes or not args.bytes):
            print("%-34s  identical values%s" % (path.name, ", identical bytes" if same_bytes else ""))
            continue
        bad += 1
        if a == b:
            print("%-34s  identical values, DIFFERENT bytes" % path.name)
            continue
        print("%-34s  DIFFERENT" % path.name)
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                print("    key %r differs" % (k,))
    print("%d of %d files differ" % (bad, len(files)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
