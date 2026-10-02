// One particle-number sector of a finite cluster, truncated by hop distance.
//
// States are occupation vectors packed four bits per site.  They are
// enumerated breadth-first from a seed set (the Mott state, or the one-doublon
// or one-holon states), so the states within hop distance d of the seeds form
// a prefix of the list.  Perturbative vectors are stored densely over such
// prefixes.
//
// Hopping uses the similarity-transformed basis in which b^dag_i b_j has the
// integer matrix element n_j (the source occupation), so every amplitude stays
// rational.
#pragma once

#include "graph.hpp"
#include "modp.hpp"

#include <boost/unordered/unordered_flat_map.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace nlce {

inline constexpr int kMaxOccupation = 15;  // four bits per site

inline int occ(u64 state, int site) { return static_cast<int>((state >> (4 * site)) & 15u); }
inline u64 site_unit(int site) { return u64(1) << (4 * site); }
u64 uniform_state(int nv, int n);

// An incoming move s -> t that carries one boson from site `from` to site
// `to` along edge `bond`.  `fwd` is T[t,s] = n_from(s); `bwd` is T[s,t] =
// n_to(t), the element of the transposed hopping.  `orient` is +1 when the
// boson moves toward the smaller vertex label of the edge.
struct Move {
  std::uint32_t s;
  std::uint8_t fwd;
  std::uint8_t bwd;
  std::uint8_t bond;
  std::int8_t orient;
};

struct Sector {
  int nv = 0;
  int nseeds = 0;
  int dmax = 0;
  std::vector<u64> states;
  std::vector<std::uint8_t> dist;
  std::vector<std::uint32_t> layer_end;  // [d] = number of states within distance d
  std::vector<std::uint32_t> move_begin;  // CSR offsets into `moves`, by target
  std::vector<Move> moves;                // sorted by source within each target
  std::vector<std::uint16_t> energy_class;  // index into `energies`
  std::vector<std::pair<int, int>> energies;  // (sum n(n-1)/2, sum_bonds (n_i-1)(n_j-1))
  std::vector<std::uint8_t> seed_dist;     // [t * nseeds + q], capped at 255
  boost::unordered_flat_map<u64, std::uint32_t> index;

  std::uint32_t len(int d) const { return d < 0 ? 0 : layer_end[std::min(d, dmax)]; }
  std::uint32_t size() const { return static_cast<std::uint32_t>(states.size()); }
};

// All states within hop distance `dmax` of `seeds`, with incoming moves
// restricted to that set.  `seed_distances` also records the distance from
// each individual seed, measured within the set.  `max_occupation` below
// kMaxOccupation selects the model in which no site holds more bosons: hops
// into a full site are dropped.  In the full model, reaching kMaxOccupation is
// an error, since the packing cannot represent more.
Sector build_sector(int nv, const std::vector<Edge>& edges, const std::vector<u64>& seeds, int dmax,
                    bool seed_distances, int max_occupation = kMaxOccupation);

}  // namespace nlce
