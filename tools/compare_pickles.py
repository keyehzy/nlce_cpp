#!/usr/bin/env python3
"""Check that regenerated series pickles equal a reference set exactly.

Compares every *_site_tri_s{N}_v*.pkl present in NEW against the file of the
same name in REF (default: validate/, the reference s = 9 and s = 10 series),
as Python objects (Fraction values, dictionary contents).  Also reports
whether the files are byte-identical.

Usage: compare_pickles.py NEW [REF]
"""

import hashlib
import pickle
import sys
from pathlib import Path


def main():
    new = Path(sys.argv[1])
    ref = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "validate"
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
        if a == b:
            print("%-34s  identical values%s" % (path.name, ", identical bytes" if same_bytes else ""))
            continue
        bad += 1
        print("%-34s  DIFFERENT" % path.name)
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                print("    key %r differs" % (k,))
    print("%d of %d files differ" % (bad, len(files)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
