# nlce_run — exact site-cluster series in C++

C++20 implementation of the site-cluster linked-cluster expansion in
`../series` (nlce.py, nlce_sq.py, nlce_chi.py): the charge gap Delta(q),
the one-body structure factor S(q), the chiral susceptibility chi_kappa and the
equal-time moment m0 of the frustrated triangular-lattice extended
Bose-Hubbard model at unit filling, as exact rational series in x = t/U.
The same expansion runs on the chain and the square, honeycomb and kagome
lattices (Delta, S and the energy; chi and m0 need the triangular staggered
current).

## Build

Requires CMake >= 3.24, a C++20 compiler, FLINT 3, GMP and Boost headers
(`brew install cmake ninja flint gmp boost`).

    cmake -S . -B build -G Ninja
    cmake --build build
    ctest --test-dir build                   # unit tests, s = 7 regression, paper and torus checks
    cmake --build build --target validate    # full s = 9 run vs validate/
    cmake --build build --target validate_paper   # vs Elstner and Monien, below

## Run

    ./build/nlce_run run --nsites 10 --out out_s10            # all six V/U
    ./build/nlce_run run --nsites 10 --v 1/5 --threads 8 --out out
    python3 tools/to_pickles.py out_s10 out_s10/pkl            # series/*.pkl format
    ./build/nlce_run run --lattice square --nsites 11 --v 0 --out out_sq

`--lattice` is `triangular` (default), `square`, `chain`, `honeycomb` or
`kagome`.  The hopping enters as +t (frustrated on the triangular and kagome
lattices), so the usual -t model follows by x -> -x.  `--nsites s` gives Delta and S through x^(s-1), chi and m0 through
x^(s-2).  `tools/q0_series.py SERIES.json` prints Delta(q=0), S(q=0) and
E/N in the -t convention.
`--dh 1` adds the doublon-holon interaction (below) through x^(s-2).
`nlce_run geometry --nsites s` prints cluster, class and key counts;
`nlce_run bench` times one cluster (see the header of `src/main.cpp`).

On an Apple M4 (10 cores, 16 GB) the six-V/U set takes about 25 seconds at
s = 9 and 6 minutes at s = 10 (1.4 GB peak), reproducing the HPC pickles in
`validate/` exactly.

## Method

* **Geometry** (`lattice`, `graph`, `geometry`): connected site clusters modulo
  translation and the point group (D6, D4, or inversion on the chain), grouped
  into graph-isomorphism classes by an individualisation-refinement canonical
  form; current patterns are canonicalised over automorphisms and a global
  sign.  The honeycomb and kagome lattices are the triangular lattice without
  its hexagon centres (a - b = 0 mod 3, resp. a and b even), with the D6 of a
  hexagon centre and the translations that preserve the sublattices; clusters
  are counted per unit cell of two or three sites and the lattice sums
  divided accordingly.
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
* **Leading orders** (`pipeline`): a term touching only some sites of a cluster
  cancels in its cumulant, so cumulants of an s-site class vanish below
  2s - m (E), max(s - 1, 2s - 2 - m) (Hp, Hh, S) and 2s - 2 - m (chi, m0),
  where m is the maximum matching of the cluster's bipartite double cover.
  Cumulants that vanish through the target order are skipped.
* **Occupation cap**: at those orders no site ever holds more than two bosons,
  so the largest clusters (and, for their subtraction, a second hierarchy of
  smaller ones) are computed in the model capped at two bosons per site,
  which drops about three quarters of their states.
* **Checks**: every cluster cumulant must vanish below its leading order; this
  is verified exactly in every lane.

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
`validate/` is empty.  The `validate` target runs s = 9 in full (about
half a minute) and requires byte-identical pickles.  Set `-DNLCE_REF_DIR=...` to use
references elsewhere.

`tools/check_cluster.py` compares per-cluster series against the Python
reference code, which it expects in `../series` (pt.py, chi.py, neutral.py):

    python3 tools/check_cluster.py --smax 6 --ng 7 --nc 6

## Verification against Elstner and Monien

N. Elstner and H. Monien, cond-mat/9905367, Tables I-III, list Delta(q=0) and
S(q=0) at V/U = 0 through x^13 for the square lattice, the triangular lattice
and the chain (in the -t convention; Appendix A adds the chain's energy through
x^6).  `tools/paper_check.py` holds those tables, runs `nlce_run` and compares
exactly; the `paper_*` ctests do so at small sizes and `validate_paper` at

| lattice    | s  | orders    | time   | peak   |
|------------|----|-----------|--------|--------|
| chain      | 14 | x^0..x^13 | 10 s   | 1.9 GB |
| square     | 11 | x^0..x^10 | 25 s   | 1.3 GB |
| triangular | 10 | x^0..x^9  | 25 s   | 1.1 GB |

Larger runs reach further: the square lattice at s = 12 (x^11, 2.5 min,
4.1 GB) and s = 13 (x^12, 46 min with 4 threads, 5 GB), the triangular
lattice at s = 11 (x^10, 5.4 min, 1.8 GB):

    python3 tools/paper_check.py --lattice square --nsites 12

Every coefficient reached agrees exactly, except where the paper itself is
not exact.  The chain's S at x^10 is printed as 22598877209/4375 where the
expansion, and an independent brute-force Rayleigh-Schroedinger calculation on
a 12-site ring, give 22598877209/4374.  The square lattice's x^12 entries are
printed as fractions but are rounded: each numerator equals our exact
coefficient times the printed denominator, rounded to an integer (agreement
to 30 digits).  Entries printed with decimals (triangular x^11 on, square S
at x^13) are compared to a relative 1e-12.

## Honeycomb lattice

No published series exists to compare with, so the honeycomb lattice is
checked by `tools/torus_check.py` (the `torus_honeycomb` ctest): a
14-site periodic torus, solved as a single cluster, bypasses the cluster
enumeration, symmetry reduction and lattice sums, and its shortest
non-contractible cycle of 6 bonds makes it exact through x^5, the first order
at which Delta sees a hexagon.  Its orders 1 and 2 also follow from the other
lattices, since below the first loop a coefficient depends on the lattice
only through the coordination number z, e.g. Delta = 1 - 3z x + (21z/2 - 4z^2)
x^2; and a unit test checks that dropping the point group leaves the q = 0
sums unchanged.

At s = 14 (20 minutes on 6 threads, 9 GB) the series at V/U = 0, in the -t
convention of the tables above and per site, are

| n | Delta(q=0) | S(q=0) | E/N |
|---:|---:|---:|---:|
| 0 | 1 | 3 | 0 |
| 1 | -9 | 24 | 0 |
| 2 | -9/2 | 216 | -6 |
| 3 | -117/2 | 2208 | 0 |
| 4 | -21513/40 | 23368 | -12 |
| 5 | -793067/200 | 256816 | 0 |
| 6 | -6829763/600 | 24510800/9 | -7880/3 |
| 7 | -2976150863/15000 | 801153440/27 | 0 |
| 8 | -32189735960147/10800000 | 26530148846/81 | -37268/3 |
| 9 | -3038973213234487/162000000 | 2303208047051/630 | 0 |
| 10 | -1850978523968117671/11340000000 | 575692849881866503/14288400 | -26083004863/3402 |
| 11 | -37317540258992730821489/32148900000000 | 168306045066538885523/375070500 | 0 |
| 12 | -845301414595631798429657071/27848984625000000 | 19856589680969707796829991/3960744480000 | -125859797456639/793800 |
| 13 | -10865364177213087797356320301799/192993463451250000000 | 257774372512885705363929216401/4574659874400000 | 0 |

The ratios of the S coefficients, about 11.2 at x^13, put the end of the
n = 1 Mott lobe near t/U = 0.085.

## Kagome lattice

The kagome lattice is checked the same way: its site clusters are counted
independently in a unit test, dropping the point group leaves the q = 0
sums unchanged, and the 15-site torus of the `torus_kagome` ctest, the best
with at most 16 sites, has non-contractible cycles of 4 bonds, so it is
exact through x^3, the first order at which the gap sees a triangle.  Orders
1 and 2 equal the square lattice's, which also has z = 4.

The kagome lattice is not bipartite, so the sign of the hopping matters.  In
the -t convention of the tables, the bottom of both the particle and the hole
band is the uniform state at q = 0, and Delta(q=0) is the Mott gap; for +t
(nlce_run's sign, x -> -x) the band bottom is the flat band instead.  At
s = 14 (1.9 hours on 5 threads, 10 GB resident and well beyond in compressed
memory) the series at V/U = 0, per site, are

| n | Delta(q=0) | S(q=0) | E/N |
|---:|---:|---:|---:|
| 0 | 1 | 3 | 0 |
| 1 | -12 | 32 | 0 |
| 2 | -22 | 432 | -8 |
| 3 | -75 | 5952 | -24 |
| 4 | -18769/10 | 260624/3 | -40 |
| 5 | -1087631/60 | 3927520/3 | -376/3 |
| 6 | -1133014847/4500 | 539721152/27 | -28640/9 |
| 7 | -31170197257/13500 | 24806797120/81 | -1706768/27 |
| 8 | -535354829732897/18900000 | 5704703275684/1215 | -7644904/9 |
| 9 | -3920056445180326307/11907000000 | 1317240246607319/18225 | -605093252/81 |
| 10 | -697318177709490684707/117209531250 | 60026911904889729341/53581500 | -2030095804936/42525 |
| 11 | -183365470329577591286207713/2362944150000000 | 2161248486412136942537093/123773265000 | -48981304920963067/107163000 |
| 12 | -2290711806389476905930658932763/2046900369937500000 | 156033605031432604882333176353/571832484300000 | -78935190877178700373/9168390000 |
| 13 | -3127435723051715146765371026090141/236416992727781250000 | 146602210124785682880061861510244833/34344259007058000000 | -4266142812217166175945449/29705583600000 |

The ratios of the S coefficients, about 15.6 at x^13, put the end of the
n = 1 Mott lobe near t/U = 0.062 for -t hopping.

## Doublon-holon interaction and the chiral exciton

`run --dh 1` also expands the two-particle sector: the Bloch effective
Hamiltonian H2 of the states with one doublon and one holon, minus its
one-particle parts,

    Idh = (H2 - E) - (Hp - E) x 1 - 1 x (Hh - E),

which is nonzero only while the two are close.  The lattice sums give, at
total momentum zero, Idh[(r', r)] from relative coordinate r = r_d - r_h to r',
summed over the translation of the pair, through x^(s-2); `tools/exciton_series.py`
adds the one-particle hopping and computes the energy of the chiral exciton,
the adjacent pair in the staggered-current pattern that the current operator
creates from the Mott state, as an exact series.  For V > 0 that state is
non-degenerate in its symmetry sector at x = 0, at energy 1 - V.  A chiral
Mott insulator appears where its energy reaches zero while the particle-hole
gap Delta(K) is still open.

Two points make the expansion cluster additive:

* The doublon-holon states have unperturbed energy 1 or 1 - V, so their wave
  operator solves the generalised Bloch equation, with one energy
  denominator per seed (`bloch` in `cluster.cpp`; the one-particle sectors
  are its special case of equal seed energies).
* The Mott ground state has components u on the doublon-holon states
  themselves, so plain Bloch coordinates of a pair in one part of a
  cluster pick up pairs from the vacuum fluctuations of another part, and the
  cumulants fail to vanish (at x^3 on six sites).  Coordinates measured from
  the dressed vacuum, c(t) = P2 t - u <vac|t>, factorise; they are a rank-one
  similarity transform of the Bloch effective Hamiltonian.

Checks: every Idh cumulant of an s-site class vanishes exactly below x^(s-2)
in every lane (s - 2 hops are needed to touch every site), and the occupation
cap again holds (unit tests through s = 7); the two-site values follow by
hand; on a 4 x 4 torus solved as one cluster, the chiral exciton energy from
H2 equals brute-force Rayleigh-Schroedinger theory in the full Hilbert
space, and away from wrap-around effects every relative-coordinate amplitude
equals the lattice expansion's; and a 6 x 6 torus by brute force gives the
lattice series' 4/5 - 6x - (68/21) x^2 + (184897/294) x^3 at V/U = 1/5.  (A 4 x 4 torus is not
exact at x^2 for V > 0: a virtual pair three sites away interacts with the
first across the wrap-around bond without hopping.)

The cost is about six times that of a plain run (s = 9: 38 s, 0.8 GB
against 7 s, 0.2 GB at one V/U; s = 10 at four V/U: 38 minutes, 4 GB).
