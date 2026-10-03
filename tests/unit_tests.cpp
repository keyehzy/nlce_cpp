// Unit tests for the arithmetic, labelling and geometry layers.  The
// end-to-end checks against the Python implementation live in tools/.
#include "cluster.hpp"
#include "driver.hpp"
#include "geometry.hpp"
#include "graph.hpp"
#include "modp.hpp"
#include "pipeline.hpp"
#include "rational.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nlce;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

void test_modulus() {
  std::mt19937_64 rng(1);
  for (u64 p : moduli(4)) {
    const Modulus m(p);
    for (int i = 0; i < 20000; ++i) {
      const u64 a = rng() % p, b = rng() % p;
      check(m.mul(a, b) == static_cast<u64>(static_cast<u128>(a) * b % p), "mul");
      const u64 x = rng();
      check(m.reduce(x) == x % p, "reduce u64");
      const u128 y = (static_cast<u128>(rng()) << 64) | rng();
      check(m.reduce(y) == static_cast<u64>(y % p), "reduce u128");
      if (a) check(m.mul(a, m.inv(a)) == 1, "inverse");
    }
    check(m.from_int(-1) == p - 1, "negative residue");
  }
}

void test_rational() {
  check(parse_rational("2/4") == Rational{1, 2}, "reduce 2/4");
  check(parse_rational("-3/6") == Rational{-1, 2}, "reduce -3/6");
  check(parse_rational("0/7") == Rational{0, 1}, "reduce 0/7");
  check(parse_rational("5") == Rational{5, 1}, "integer");
  for (const char* bad : {"1/0", "1/-5", "1/5x", "abc", "", "/3"}) {
    bool threw = false;
    try {
      parse_rational(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, std::string("reject \"") + bad + "\"");
  }
  check(to_string(Rational{3, 20}) == "3/20" && to_string(Rational{0, 1}) == "0", "to_string");
  check(file_tag(Rational{3, 20}) == "3over20" && file_tag(Rational{0, 1}) == "0", "file_tag");
}

std::vector<Edge> random_graph(std::mt19937& rng, int n, double density) {
  std::vector<Edge> edges;
  std::bernoulli_distribution coin(density);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (coin(rng)) edges.emplace_back(i, j);
    }
  }
  return edges;
}

void test_canonical_form() {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 400; ++trial) {
    const int n = 2 + trial % 9;
    const auto edges = random_graph(rng, n, 0.45);
    const Canon base = canonical_form(n, edges);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<Edge> relabelled;
    for (auto [i, j] : edges) relabelled.emplace_back(std::min(perm[i], perm[j]), std::max(perm[i], perm[j]));
    const Canon other = canonical_form(n, relabelled);
    check(base.cert == other.cert, "certificate invariant under relabelling");

    // The canonical map must carry the graph onto the certificate's graph.
    std::vector<Edge> mapped;
    for (auto [i, j] : edges) mapped.emplace_back(std::min(base.perm[i], base.perm[j]), std::max(base.perm[i], base.perm[j]));
    std::sort(mapped.begin(), mapped.end());
    check(mapped == edges_from_cert(base.cert), "canonical map reproduces certificate");

    const auto autos = automorphisms(n, edges);
    check(!autos.empty(), "identity automorphism");
    for (const auto& a : autos) {
      std::vector<Edge> img;
      for (auto [i, j] : edges) img.emplace_back(std::min(a[i], a[j]), std::max(a[i], a[j]));
      std::sort(img.begin(), img.end());
      check(img == edges, "automorphism preserves edges");
    }
  }
  // A path and a star on four vertices are not isomorphic.
  const Canon path = canonical_form(4, {{0, 1}, {1, 2}, {2, 3}});
  const Canon star = canonical_form(4, {{0, 1}, {0, 2}, {0, 3}});
  check(!(path.cert == star.cert), "path vs star");
  check(automorphisms(6, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {0, 5}}).size() == 12, "hexagon automorphisms");
}

void test_double_cover_matching() {
  check(double_cover_matching(1, {}) == 0, "single vertex");
  check(double_cover_matching(2, {{0, 1}}) == 2, "edge");
  check(double_cover_matching(3, {{0, 1}, {1, 2}}) == 2, "path P3");
  check(double_cover_matching(3, {{0, 1}, {1, 2}, {0, 2}}) == 3, "triangle");
  check(double_cover_matching(4, {{0, 1}, {0, 2}, {0, 3}}) == 2, "star K1,3");
  check(double_cover_matching(5, {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}}) == 5, "triangle with a tail of two");
  check(double_cover_matching(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}) == 4, "path P5");
}

// Reference counts from the Python implementation (series/nlce.py,
// series/nlce_chi.py): lattice clusters, isomorphism classes and decorated
// keys per size.
void test_geometry() {
  const int clusters[] = {0, 1, 1, 3, 7, 22, 82, 333, 1448, 6572};
  const int classes[] = {0, 1, 1, 2, 4, 8, 22, 54, 156, 457};
  const int keys[] = {0, 1, 1, 3, 6, 17, 53, 178, 633, 2385};
  const Geometry geo = build_geometry(triangular_lattice(), 9);
  for (int s = 1; s <= 9; ++s) {
    int nc = 0, nk = 0;
    for (const auto& c : geo.classes) nc += c.nv == s;
    for (const auto& k : geo.keys) nk += geo.classes[k.cls].nv == s;
    check(geo.cluster_counts[s] == clusters[s], "cluster count s=" + std::to_string(s));
    check(nc == classes[s], "class count s=" + std::to_string(s));
    check(nk == keys[s], "key count s=" + std::to_string(s));
  }
  // Embeddings per site of each size sum to the number of fixed clusters
  // divided by the site count, i.e. the lattice-animal series per site.
  std::int64_t pairs = 0;
  for (const auto& c : geo.classes) {
    if (c.nv == 3) pairs += c.mult;
  }
  check(pairs == 11, "3-site embeddings per site");
}

// Chain and square lattice: clusters modulo symmetry (free polyominoes on the
// square lattice, OEIS A000105) and embeddings per site (fixed polyominoes,
// A001168).  Neither lattice carries the staggered current.
void test_bipartite_geometry() {
  const Geometry chain = build_geometry(chain_lattice(), 8);
  for (int s = 1; s <= 8; ++s) {
    std::int64_t mult = 0;
    for (const auto& c : chain.classes) mult += c.nv == s ? c.mult : 0;
    check(chain.cluster_counts[s] == 1 && mult == 1, "chain clusters s=" + std::to_string(s));
  }
  check(chain.classes.size() == 8 && chain.keys.empty(), "chain classes and keys");
  check(chain.displacements.size() == 8, "chain displacements");

  const int free[] = {0, 1, 1, 2, 5, 12, 35, 108, 369};
  const std::int64_t fixed[] = {0, 1, 2, 6, 19, 63, 216, 760, 2725};
  const Geometry square = build_geometry(square_lattice(), 8);
  for (int s = 1; s <= 8; ++s) {
    std::int64_t mult = 0;
    for (const auto& c : square.classes) mult += c.nv == s ? c.mult : 0;
    check(square.cluster_counts[s] == free[s], "square clusters s=" + std::to_string(s));
    check(mult == fixed[s], "square embeddings per site s=" + std::to_string(s));
  }
  check(square.keys.empty(), "square keys");
  // Summed over every displacement of its orbit, each class contributes all
  // nv^2 site pairs of each of its embeddings.
  for (const auto& c : square.classes) {
    std::int64_t pairs = 0;
    for (const auto& e : c.embeddings) {
      pairs += e.fac * displacement_orbit_size(square_lattice(), square.displacements[e.cd]);
    }
    check(pairs == c.mult * c.nv * c.nv * 8, "square embedding weights");
  }
}

// Honeycomb site clusters are polyiamonds: modulo symmetry the free ones
// (A000577), per unit cell the fixed ones (A001420).
void test_honeycomb_geometry() {
  const int free[] = {0, 1, 1, 1, 3, 4, 12, 24, 66, 160, 448};
  const std::int64_t fixed[] = {0, 2, 3, 6, 14, 36, 94, 250, 675, 1838, 5053};
  const Lattice& lat = honeycomb_lattice();
  check(lat.cell_sites() == 2, "honeycomb unit cell");
  const Geometry geo = build_geometry(lat, 10);
  for (int s = 1; s <= 10; ++s) {
    std::int64_t mult = 0;
    for (const auto& c : geo.classes) mult += c.nv == s ? c.mult : 0;
    check(geo.cluster_counts[s] == free[s], "honeycomb clusters s=" + std::to_string(s));
    check(mult == fixed[s], "honeycomb embeddings per cell s=" + std::to_string(s));
  }
  for (const auto& c : geo.classes) {
    for (auto [i, j] : c.edges) check(i != j, "honeycomb edge");
    for (int v = 0; v < c.nv; ++v) {
      int deg = 0;
      for (auto [i, j] : c.edges) deg += (i == v) + (j == v);
      check(deg <= 3, "honeycomb degree");
    }
  }
}

// Kagome site clusters, counted independently from the kagome as the edge
// midpoints of the triangular lattice of hexagon centres: modulo symmetry and
// per unit cell.
void test_kagome_geometry() {
  const int free[] = {0, 1, 1, 3, 4, 12, 27, 78, 208, 635};
  const std::int64_t fixed[] = {0, 3, 6, 14, 36, 99, 281, 816, 2415, 7260};
  const Lattice& lat = kagome_lattice();
  check(lat.cell_sites() == 3, "kagome unit cell");
  const Geometry geo = build_geometry(lat, 9);
  for (int s = 1; s <= 9; ++s) {
    std::int64_t mult = 0;
    for (const auto& c : geo.classes) mult += c.nv == s ? c.mult : 0;
    check(geo.cluster_counts[s] == free[s], "kagome clusters s=" + std::to_string(s));
    check(mult == fixed[s], "kagome embeddings per cell s=" + std::to_string(s));
  }
  for (const auto& c : geo.classes) {
    for (int v = 0; v < c.nv; ++v) {
      int deg = 0;
      for (auto [i, j] : c.edges) deg += (i == v) + (j == v);
      check(deg <= 4, "kagome degree");
    }
  }
}

// Per lane: Delta(q=0), S(q=0), E/N, chi and m0 residues, i.e. the
// displacement sums weighted by orbit size.
std::vector<std::vector<u64>> q0_sums(const Geometry& geo, const std::vector<std::vector<u64>>& series,
                                      const std::vector<LaneSpec>& lanes, int ng, int nc) {
  const SeriesLayout lay{ng, nc, static_cast<int>(geo.displacements.size())};
  std::vector<std::vector<u64>> out;
  for (std::size_t g = 0; g < series.size(); ++g) {
    const Modulus m(lanes[g].p);
    const auto& x = series[g];
    std::vector<u64> sums(3 * (ng + 1) + 2 * (nc + 1), 0);
    for (int cd = 0; cd < lay.ncd; ++cd) {
      const Site d = geo.displacements[cd];
      const u64 w = m.from_int(displacement_orbit_size(*geo.lattice, d));
      for (int k = 0; k <= ng; ++k) {
        const std::size_t at = static_cast<std::size_t>(cd) * (ng + 1) + k;
        sums[k] = m.add(sums[k], m.mul(w, m.add(x[lay.hp() + at], x[lay.hh() + at])));
        if (!(d == Site{})) sums[ng + 1 + k] = m.add(sums[ng + 1 + k], m.mul(w, x[lay.s() + at]));
      }
    }
    for (int k = 0; k <= ng; ++k) sums[2 * (ng + 1) + k] = x[lay.en() + k];
    for (int k = 0; k <= nc; ++k) {
      sums[3 * (ng + 1) + k] = x[lay.chi() + k];
      sums[3 * (ng + 1) + nc + 1 + k] = x[lay.m0() + k];
    }
    out.push_back(std::move(sums));
  }
  return out;
}

// Classifying clusters modulo the point group must not change the q = 0
// sums: the full group and the identity alone give the same series.
void test_point_group_reduction() {
  std::vector<LaneSpec> lanes;
  for (u64 p : moduli(2)) {
    lanes.push_back({{0, 1}, p});
    lanes.push_back({{1, 5}, p});
  }
  PassOptions opts;
  opts.threads = 2;
  opts.verbose = false;
  for (const auto& [base, smax] : {std::pair{&honeycomb_lattice(), 8}, std::pair{&kagome_lattice(), 7},
                                   std::pair{&triangular_lattice(), 6}}) {
    Lattice bare = *base;
    bare.group = {{1, 0, 0, 1}};
    const int ng = smax - 1, nc = base->currents ? smax - 2 : 0;
    const Geometry full = build_geometry(*base, smax);
    const Geometry plain = build_geometry(bare, smax);
    check(plain.classes.size() == full.classes.size(), base->name + " classes without point group");
    const auto a = q0_sums(full, run_pass(full, ng, nc, lanes, opts), lanes, ng, nc);
    const auto b = q0_sums(plain, run_pass(plain, ng, nc, lanes, opts), lanes, ng, nc);
    check(a == b, base->name + " q = 0 sums without point group");
  }
}

// A small full run must satisfy the exact cumulant cancellation in every lane;
// run_pass throws otherwise.
void test_small_pass() {
  const Geometry geo = build_geometry(triangular_lattice(), 5);
  std::vector<LaneSpec> lanes;
  for (u64 p : moduli(3)) {
    lanes.push_back({{0, 1}, p});
    lanes.push_back({{1, 5}, p});
  }
  PassOptions opts;
  opts.threads = 2;
  opts.verbose = false;
  bool ok = true;
  try {
    run_pass(geo, 4, 3, lanes, opts);
  } catch (const std::exception& e) {
    ok = false;
    std::fprintf(stderr, "%s\n", e.what());
  }
  check(ok, "cumulant cancellation, s<=5");
}

// The largest classes computed with at most two bosons per site must give the
// lattice series of the full model exactly.
void test_occupation_cap() {
  const Geometry geo = build_geometry(triangular_lattice(), 6);
  std::vector<LaneSpec> lanes;
  for (u64 p : moduli(2)) {
    lanes.push_back({{0, 1}, p});
    lanes.push_back({{1, 5}, p});
    lanes.push_back({{3, 20}, p});
  }
  PassOptions opts;
  opts.threads = 2;
  opts.verbose = false;
  const auto capped = run_pass(geo, 5, 4, lanes, opts);
  opts.cap_largest = false;
  const auto full = run_pass(geo, 5, 4, lanes, opts);
  check(capped == full, "occupation-capped largest classes, s<=6");
}

// Doublon-holon interaction.  On two sites at V/U = 1/4 an adjacent pair
// gains -V at x^0; at x^2 it annihilates into the Mott state, which returns it
// in place or swapped (2 x^2 / (1 - V) each), against the one-particle
// self-energies and the Mott energy.  On larger clusters the interaction must
// be cluster additive (run_pass checks every cumulant below x^(s-2)), and the
// largest classes may again be computed with two bosons per site.
void test_doublon_holon() {
  ClusterInput in{2, {{0, 1}}, {}, 2, 0};
  in.ndh = 2;
  const ClusterSeries two = cluster_series(plan_cluster(in), Rational{1, 4}, 2);
  // Rows ((i'*2 + j')*2 + i)*2 + j after E, Hp, Hh, corr (4 pairs each), chi, m0.
  const std::size_t base = 3 + 3 * 4 * 3;
  auto idh = [&](int row, int k) { return two.rec.values[base + row * 3 + k]; };
  check(idh(5, 0) == "-1/4" && idh(5, 1) == "0" && idh(5, 2) == "-2/3", "adjacent doublon-holon pair, two sites");
  check(idh(6, 0) == "0" && idh(6, 2) == "8/3", "doublon-holon swap, two sites");

  Geometry geo = build_geometry(triangular_lattice(), 7);
  build_dh_keys(geo);
  std::vector<LaneSpec> lanes;
  for (u64 p : moduli(2)) {
    lanes.push_back({{0, 1}, p});
    lanes.push_back({{1, 4}, p});
    lanes.push_back({{3, 10}, p});
  }
  PassOptions opts;
  opts.threads = 2;
  opts.verbose = false;
  opts.ndh = 5;
  std::vector<std::vector<u64>> capped, full;
  bool ok = true;
  try {
    capped = run_pass(geo, 6, 5, lanes, opts);
    opts.cap_largest = false;
    full = run_pass(geo, 6, 5, lanes, opts);
  } catch (const std::exception& e) {
    ok = false;
    std::fprintf(stderr, "%s\n", e.what());
  }
  check(ok, "doublon-holon cumulant cancellation, s<=7");
  check(ok && capped == full, "doublon-holon interaction with occupation-capped largest classes, s<=7");
}

// Inputs beyond the overflow budget must be rejected, not computed wrongly.
void test_limits() {
  bool threw = false;
  try {
    plan_cluster({2, {{0, 1}}, {}, kMaxOrder + 1, 1});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "order beyond kMaxOrder rejected");
}

}  // namespace

int main() {
  test_modulus();
  test_rational();
  test_canonical_form();
  test_double_cover_matching();
  test_geometry();
  test_bipartite_geometry();
  test_honeycomb_geometry();
  test_kagome_geometry();
  test_point_group_reduction();
  test_small_pass();
  test_occupation_cap();
  test_doublon_holon();
  test_limits();
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all unit tests passed\n");
  return 0;
}
