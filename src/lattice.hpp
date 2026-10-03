// Lattices, their point groups, and connected site clusters.
//
// Sites are integer pairs (a, b) at a*e1 + b*e2.  The chain uses e1 alone
// (b = 0), the square lattice e1 = (1, 0), e2 = (0, 1), and the triangular
// lattice e1 = (1, 0), e2 = (1/2, sqrt3/2).  The honeycomb lattice is the
// triangular lattice without the sublattice (2a + b) = 0 (mod 3) of hexagon
// centres, so it has two sites per unit cell.  A site cluster is a connected
// set of sites carrying every lattice bond between its members; clusters are
// classified modulo translation and the point group.
#pragma once

#include <compare>
#include <optional>
#include <string>
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

// Integer 2x2 matrix acting on (a, b).
struct PointOp {
  int m0, m1, m2, m3;
  Site operator()(Site s) const { return {m0 * s.a + m1 * s.b, m2 * s.a + m3 * s.b}; }
  bool operator==(const PointOp&) const = default;
};

struct Lattice {
  std::string name;            // "chain", "square", "triangular" or "honeycomb"
  std::vector<Site> dirs;      // positive bond directions, between two sites
  std::vector<PointOp> group;  // point group about the origin, identity first
  // Sublattices: the residue of (a, b) is (form.a * a + form.b * b) mod
  // period.  Translations are the vectors of residue 0; anchors[r] is the
  // site of residue r to which a cluster's smallest site is translated, or
  // empty where residue r holds no sites.  Bravais lattices have period 1.
  int period = 1;
  Site form{};
  std::vector<std::optional<Site>> anchors{Site{}};
  bool currents = false;  // three-sublattice staggered current (chi, m0) defined

  int residue(Site s) const { return ((form.a * s.a + form.b * s.b) % period + period) % period; }
  bool is_site(Site s) const { return anchors[residue(s)].has_value(); }
  int cell_sites() const;
};

const Lattice& chain_lattice();
const Lattice& square_lattice();
const Lattice& triangular_lattice();
const Lattice& honeycomb_lattice();

// Throws std::invalid_argument for an unknown name.
const Lattice& lattice_by_name(const std::string& name);

// Lexicographically smallest point-group image of a displacement.
Site canonical_displacement(const Lattice& lat, Site d);
int displacement_orbit_size(const Lattice& lat, Site d);

// Three-sublattice label l = (2a + b) mod 3 of the triangular lattice and the
// staggered current sign of the oriented bond p -> q: +1 if
// l(q) = l(p) + 1 (mod 3), else -1.
int sublattice(Site s);
int current_sign(Site p, Site q);

using Bond = std::pair<Site, Site>;

// Every lattice bond with both endpoints in `sites`, each as (smaller,
// larger) and the list sorted.
std::vector<Bond> induced_bonds(const Lattice& lat, const std::vector<Site>& sites);

// Canonical representative: translated so its smallest site is the anchor of
// its residue, minimised over the point group, sorted.
std::vector<Site> canonical_sites(const Lattice& lat, const std::vector<Site>& sites);

// Number of distinct translation classes in the point-group orbit.
int site_orbit_size(const Lattice& lat, const std::vector<Site>& sites);

// reps[s] lists the canonical connected clusters of s sites, s = 1..smax.
// Each size is grown from the previous one by adding a neighbouring site;
// the order of discovery is deterministic.
std::vector<std::vector<std::vector<Site>>> enumerate_site_clusters(const Lattice& lat, int smax);

}  // namespace nlce
