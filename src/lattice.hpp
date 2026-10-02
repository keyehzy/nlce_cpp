// Triangular lattice, its D6 point group, and connected site clusters.
//
// Sites are integer pairs (a, b) at a*e1 + b*e2 with e1 = (1, 0) and
// e2 = (1/2, sqrt3/2).  A site cluster is a connected set of sites carrying
// every lattice bond between its members; clusters are classified modulo
// translation and the point group.
#pragma once

#include <array>
#include <compare>
#include <utility>
#include <vector>

namespace nlce {

struct Site {
  int a = 0;
  int b = 0;
  auto operator<=>(const Site&) const = default;
};

inline Site operator+(Site s, Site t) { return {s.a + t.a, s.b + t.b}; }
inline Site operator-(Site s, Site t) { return {s.a - t.a, s.b - t.b}; }

// Positive bond directions.
inline constexpr std::array<Site, 3> kDirs = {{{1, 0}, {0, 1}, {1, -1}}};

// Integer 2x2 matrix acting on (a, b).
struct PointOp {
  int m0, m1, m2, m3;
  Site operator()(Site s) const { return {m0 * s.a + m1 * s.b, m2 * s.a + m3 * s.b}; }
};

const std::array<PointOp, 12>& point_group();

// Lexicographically smallest point-group image of a displacement.
Site canonical_displacement(Site d);
int displacement_orbit_size(Site d);

// Three-sublattice label l = (2a + b) mod 3 and the staggered current sign of
// the oriented bond p -> q: +1 if l(q) = l(p) + 1 (mod 3), else -1.
int sublattice(Site s);
int current_sign(Site p, Site q);

using Bond = std::pair<Site, Site>;

// Every lattice bond with both endpoints in `sites`, each as (smaller,
// larger) and the list sorted.
std::vector<Bond> induced_bonds(const std::vector<Site>& sites);

// Canonical representative: translated so its smallest site is the origin,
// minimised over the point group, sorted.
std::vector<Site> canonical_sites(const std::vector<Site>& sites);

// Number of distinct translation classes in the point-group orbit.
int site_orbit_size(const std::vector<Site>& sites);

// reps[s] lists the canonical connected clusters of s sites, s = 1..smax.
// Each size is grown from the previous one by adding a neighbouring site;
// the order of discovery is deterministic.
std::vector<std::vector<std::vector<Site>>> enumerate_site_clusters(int smax);

}  // namespace nlce
