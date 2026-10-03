// Site-cluster linked-cluster expansion modulo primes.
//
// Classes are processed in waves of increasing size.  Within a wave every
// class is independent: its finite-cluster series are computed, the weights
// of its connected induced subclusters (all from earlier waves) subtracted,
// and the resulting cumulant added to the lattice sums with its embedding
// multiplicities.  The cumulant of an s-site class must vanish below x^(s-1)
// (x^(s-2) for chi and m0); this is checked exactly in every lane, and only
// the non-vanishing orders are kept for later subtraction.
#pragma once

#include "geometry.hpp"
#include "modp.hpp"

#include <vector>

namespace nlce {

// Flattened lattice series of one lane:
//   EN[k], Hp[cd][k], Hh[cd][k], S[cd][k], chi[k'], m0[k'], Idh[key][k''],
//   Hp_pairs[pc][k], Hh_pairs[pc][k]
// with k <= ng, k' <= nc, k'' <= ndh, cd over Geometry::displacements and key
// over Geometry::dh_keys.  Idh[(r', r)] is the doublon-holon interaction at
// total momentum zero, from relative coordinate r to r', summed over the
// translation of the pair.  Hp_pairs[pc] and Hh_pairs[pc] are the amplitudes
// between the two sites of any pair of class pc (Geometry::pairs).
struct SeriesLayout {
  int ng = 0, nc = 0, ncd = 0;
  int ndh = -1, nkdh = 0;
  int npc = 0;  // pair classes, Geometry::pairs
  std::size_t en() const { return 0; }
  std::size_t hp() const { return en() + (ng + 1); }
  std::size_t hh() const { return hp() + static_cast<std::size_t>(ncd) * (ng + 1); }
  std::size_t s() const { return hh() + static_cast<std::size_t>(ncd) * (ng + 1); }
  std::size_t chi() const { return s() + static_cast<std::size_t>(ncd) * (ng + 1); }
  std::size_t m0() const { return chi() + (nc + 1); }
  std::size_t dh() const { return m0() + (nc + 1); }
  std::size_t hp_pairs() const { return dh() + (ndh >= 0 ? static_cast<std::size_t>(nkdh) * (ndh + 1) : 0); }
  std::size_t hh_pairs() const { return hp_pairs() + static_cast<std::size_t>(npc) * (ng + 1); }
  std::size_t size() const { return hh_pairs() + static_cast<std::size_t>(npc) * (ng + 1); }

  // Perturbative order of coefficient i.
  int order(std::size_t i) const {
    if (i < hp()) return static_cast<int>(i);
    if (i < chi()) return static_cast<int>((i - hp()) % (ng + 1));
    if (i < dh()) return static_cast<int>(i < m0() ? i - chi() : i - m0());
    if (i < hp_pairs()) return static_cast<int>((i - dh()) % (ndh + 1));
    return static_cast<int>((i - hp_pairs()) % (ng + 1));
  }
};

struct PassOptions {
  int threads = 1;
  bool verbose = true;
  bool cap_largest = true;  // largest classes in the occupation-capped model (exact; see pipeline.cpp)
  int ndh = -1;             // order of the doublon-holon interaction; negative: off (needs build_dh_keys)
};

// Returns, per lane, the lattice series residues in SeriesLayout order.
std::vector<std::vector<u64>> run_pass(const Geometry& geo, int ng, int nc, const std::vector<LaneSpec>& lanes,
                                       const PassOptions& opts);

}  // namespace nlce
