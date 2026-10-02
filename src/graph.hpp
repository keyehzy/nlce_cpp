// Canonical labelling of small simple graphs (at most 16 vertices).
//
// Two clusters with isomorphic bond graphs have identical finite-cluster
// series, so every cluster quantity is computed once per isomorphism class
// and indexed by canonical vertex labels.
#pragma once

#include "modp.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace nlce {

inline constexpr int kMaxVertices = 16;

using Edge = std::pair<int, int>;

// Upper-triangular adjacency bit string under the canonical labelling; the
// first pair (0,1) is the most significant bit, so integer order equals
// lexicographic order of the bit string.
struct Cert {
  int n = 0;
  u128 bits = 0;
  bool operator==(const Cert&) const = default;
};

struct CertHash {
  std::size_t operator()(const Cert& c) const;
};

struct Canon {
  Cert cert;
  std::vector<int> perm;  // perm[v] = canonical index of vertex v
};

// Individualisation-refinement over the full search tree; the certificate is
// the minimum over all leaves, so it is a complete isomorphism invariant.
Canon canonical_form(int n, const std::vector<Edge>& edges);

// Edges (i, j), i < j, of the canonical graph, in increasing order.
std::vector<Edge> edges_from_cert(const Cert& cert);

// Every vertex permutation preserving the edge set.
std::vector<std::vector<int>> automorphisms(int n, const std::vector<Edge>& edges);

}  // namespace nlce
