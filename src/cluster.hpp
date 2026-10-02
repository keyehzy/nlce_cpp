// Finite-cluster Rayleigh-Schroedinger series, evaluated modulo primes.
//
//   H = H0 + x T,   H0 = (1/2) sum_i n_i(n_i-1) + v sum_<ij> (n_i-1)(n_j-1)
//
// For one cluster graph and one block of (V/U, prime) lanes this computes
//
//   E        Mott ground-state energy                         through x^ng
//   Hp, Hh   one-doublon / one-holon Bloch effective
//            Hamiltonians minus E (Gelfand 1996)               through x^ng
//   corr     <b^dag_u b_w + b^dag_w b_u> for all u != w       through x^ng
//   chi, m0  per current pattern K: the g^2 energy coefficient
//            for H + g K, and <-K^2>                           through x^nc
//
// Expectation values use the left (phi) and right (psi) ground-state
// eigenvectors of the similarity-transformed Hamiltonian,
// <A> = <phi|A|psi>/<phi|psi>.  Since K is odd under time reversal and
// H(g)^T = H(-g) in the physical basis, the g^2 coefficient needs only the
// first-order response psi_g:  E_gg/2 = <phi|K|psi_g>/<phi|psi>.
//
// HOP-DISTANCE PRUNING.  A component at hop distance d from the seed states in
// the order-k vector first influences an energy coefficient at order k + d.
// Vectors are therefore truncated to d <= (target order) - k, plus the reach of
// any operator they are later contracted with.  The truncation discards only
// components that cannot contribute, so the series are exact.
#pragma once

#include "geometry.hpp"
#include "modp.hpp"
#include "sector.hpp"

#include <vector>

namespace nlce {

struct ClusterInput {
  int nv = 0;
  std::vector<Edge> edges;
  std::vector<Pattern> patterns;
  int ng = 0;  // order of E, Hp, Hh, corr
  int nc = 0;  // order of chi, m0
};

struct PairMove {
  std::uint32_t s2;
  std::uint16_t pair;  // u * nv + w with u < w
  std::uint8_t coef;   // occupation of the site the boson leaves
};

// Everything about a cluster that does not depend on V/U or the prime.
struct ClusterPlan {
  ClusterInput in;
  int diameter = 0;
  std::vector<int> graph_dist;  // nv * nv
  std::vector<int> lim_gs;      // [k] max hop distance kept in ground-state vectors
  Sector mott, particle, hole;
  std::vector<std::uint32_t> pair_begin;
  std::vector<PairMove> pairs;  // b^dag_u b_w moves over Mott states
};

ClusterPlan plan_cluster(ClusterInput in);

// Residues for one lane block.  Layouts, lane fastest:
//   E      [k][l]
//   Hp, Hh, corr  [(u*nv + w)][k][l]
//   chi, m0       [pattern][k][l]
struct RawSeries {
  int nv = 0, ng = 0, nc = 0, npat = 0;
  std::vector<u64> E, Hp, Hh, corr, chi, m0;

  std::size_t pair_at(int u, int w, int k) const {
    return ((static_cast<std::size_t>(u) * nv + w) * (ng + 1) + k) * kLanes;
  }
  std::size_t pattern_at(int q, int k) const {
    return (static_cast<std::size_t>(q) * (nc + 1) + k) * kLanes;
  }

  // Every residue of lane l: E, Hp, Hh, corr, chi, m0 in the orders above.
  std::vector<u64> lane(int l) const;
};

void compute_block(const ClusterPlan& plan, const LaneBlock& lanes, RawSeries& out);

}  // namespace nlce
