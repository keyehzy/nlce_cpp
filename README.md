# nlce_run — exact site-cluster series in C++

C++20 implementation of the site-cluster linked-cluster expansion in
`../series` (nlce.py, nlce_sq.py, nlce_chi.py): the charge gap Delta(q),
the one-body structure factor S(q), the chiral susceptibility chi_kappa and the
equal-time moment m0 of the frustrated triangular-lattice extended
Bose-Hubbard model at unit filling, as exact rational series in x = t/U.

## Build

Requires CMake >= 3.24, a C++20 compiler, FLINT 3, GMP and Boost headers
(`brew install cmake ninja flint gmp boost`).

    cmake -S . -B build -G Ninja
    cmake --build build
    ctest --test-dir build                   # unit tests + s = 7 regression
    cmake --build build --target validate    # full s = 9 run vs validate/

## Run

    ./build/nlce_run run --nsites 10 --out out_s10            # all six V/U
    ./build/nlce_run run --nsites 10 --v 1/5 --threads 8 --out out
    python3 tools/to_pickles.py out_s10 out_s10/pkl            # series/*.pkl format

`--nsites s` gives Delta and S through x^(s-1), chi and m0 through x^(s-2).
`nlce_run geometry --nsites s` prints cluster, class and key counts;
`nlce_run bench` times one cluster (see the header of `src/main.cpp`).

On an Apple M4 (10 cores, 16 GB) the six-V/U set takes about 1 minute at
s = 9 and 14 minutes at s = 10 (2.1 GB peak), reproducing the HPC pickles in
`validate/` exactly.

## Method

* **Geometry** (`lattice`, `graph`, `geometry`): connected site clusters modulo
  translation and D6, grouped into graph-isomorphism classes by an
  individualisation-refinement canonical form; current patterns are
  canonicalised over automorphisms and a global sign.
* **Exact arithmetic by multiple primes** (`modp`, `reconstruct`): every step —
  perturbation theory, subcluster subtraction, lattice assembly — is ring
  arithmetic, so it runs modulo 50-bit primes.  The lattice coefficients are
  recovered by Chinese remaindering and rational reconstruction; two extra
  primes verify every coefficient.  `run` adds primes per V/U until the
  reconstruction verifies (V/U = 0 needs a few, V/U = 1/20 at s = 10 about 35).
* **Lanes**: (V/U, prime) pairs share every V-independent structure of a
  cluster and are processed eight at a time.
* **Per-cluster series** (`sector`, `cluster`): breadth-first bases by hop
  distance, vectors truncated to the distance that can still reach the target
  order (exact pruning), left and right ground-state chains shared by E, S,
  chi and m0, chi from the first-order response only.
* **Checks**: every cluster cumulant must vanish below x^(s-1) (x^(s-2) for chi,
  m0); this is verified exactly in every lane.

## Verification against the Python code

`validate/` holds the reference series produced by the original Python
implementation on HPC: the 48 `{res,sq,chi,m0}_site_tri_s{9,10}_v*.pkl` files
for V/U = 0, 1/20, 1/10, 3/20, 1/5, 1/4.  It is not tracked by git; populate
it from the Python project with

    mkdir -p validate && cp ../series/*_site_tri_s{9,10}_*.pkl validate/

    python3 tools/to_pickles.py out_s10 out_s10/pkl
    python3 tools/compare_pickles.py --bytes out_s10/pkl        # lattice series vs validate/

Two checks run from the build directory.  An s-site expansion is exact
through x^(s-1) (x^(s-2) for chi and m0), so `tools/regression.py`, the
`regression` ctest, runs s = 7 in under a second and requires its series to
equal the leading coefficients of the s = 9 references; it is skipped when
`validate/` is empty.  The `validate` target runs s = 9 in full (about a
minute) and requires byte-identical pickles.  Set `-DNLCE_REF_DIR=...` to use
references elsewhere.

`tools/check_cluster.py` compares per-cluster series against the Python
reference code, which it expects in `../series` (pt.py, chi.py, neutral.py):

    python3 tools/check_cluster.py --smax 6 --ng 7 --nc 6
