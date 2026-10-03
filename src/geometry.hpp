// Site-cluster geometry of a lattice, independent of V/U.
//
// A class is a graph-isomorphism class of connected induced site clusters; a
// key is a class together with the canonical staggered-current pattern on its
// edges.  The gap, S(q) and energy are assembled from classes, chi and m0
// from keys.  Lattices without the staggered current have no keys.
#pragma once

#include "graph.hpp"
#include "lattice.hpp"

#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nlce {

using Pattern = std::vector<std::int8_t>;  // one sign per canonical edge

struct Embedding {
  int cd;              // index into Geometry::displacements
  int a, b;            // canonical vertex indices
  std::int64_t fac;    // embedding weight per unit cell, times the point-group order
};

// One lattice realisation of a class, modulo translation and the point group.
struct Realization {
  int orbit = 0;          // translation classes in its point-group orbit
  std::vector<Site> pos;  // pos[a] = site of canonical vertex a
};

struct ClassInfo {
  Cert cert;
  int nv = 0;
  std::vector<Edge> edges;  // canonical, sorted
  std::vector<int> edge_index;  // nv*nv, -1 where no edge
  std::vector<std::vector<int>> autos;
  int matching = 0;       // double_cover_matching of the class graph
  std::int64_t mult = 0;  // embeddings per unit cell
  std::vector<Embedding> embeddings;  // site-pair embeddings by displacement
  std::vector<int> keys;  // decorated keys of this class
  std::vector<Realization> realizations;
};

struct KeyInfo {
  int cls = 0;
  Pattern pattern;
  std::int64_t mult = 0;
};

struct Geometry {
  const Lattice* lattice = nullptr;
  int smax = 0;
  std::vector<int> cluster_counts;  // [s], lattice clusters modulo symmetry
  std::vector<ClassInfo> classes;
  std::vector<KeyInfo> keys;
  std::vector<Site> displacements;  // canonical displacements, first-seen order
  boost::unordered_flat_map<Cert, int, CertHash> class_of;
  boost::unordered_flat_map<std::string, int> key_of;
  // Doublon-holon keys (r', r), relative coordinates doublon - holon after and
  // before a process, modulo the point group acting on both; see build_dh_keys.
  std::vector<std::array<Site, 2>> dh_keys;
  std::vector<int> dh_key_orbit;  // images of each key under the point group
  boost::unordered_flat_map<std::array<int, 4>, int, boost::hash<std::array<int, 4>>> dh_key_of;

  int key_index(int cls, const Pattern& pattern) const;
  int dh_key(Site after, Site before) const;
};

Geometry build_geometry(const Lattice& lat, int smax);

// Indexes every pair of nonzero displacements within a realisation.  Defined
// on Bravais lattices only, where a displacement fixes the pair of sites up
// to translation.
void build_dh_keys(Geometry& geo);

// Lexicographically smallest image of `signs` (one per class edge, oriented
// i < j) over the class automorphisms and a global sign flip.
Pattern canonical_pattern(const ClassInfo& cls, const Pattern& signs);

// A connected proper induced subcluster of a class, mapped to its own class.
struct SubCluster {
  int cls = 0;
  std::vector<int> verts;  // parent vertex labels, increasing
  std::vector<int> map;    // map[k] = canonical index of verts[k] in `cls`
  std::vector<int> parent_edges;  // indices of parent edges inside `verts`
};

std::vector<SubCluster> subclusters(const Geometry& geo, int cls);

// Decorated key of a subcluster given the parent's pattern.
int subcluster_key(const Geometry& geo, int parent_cls, const SubCluster& sub, const Pattern& parent);

}  // namespace nlce
