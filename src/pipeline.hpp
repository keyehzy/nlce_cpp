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
//   EN[k], Hp[cd][k], Hh[cd][k], S[cd][k], chi[k'], m0[k']
// with k <= ng, k' <= nc, cd over Geometry::displacements.
struct SeriesLayout {
  int ng = 0, nc = 0, ncd = 0;
  std::size_t en() const { return 0; }
  std::size_t hp() const { return en() + (ng + 1); }
  std::size_t hh() const { return hp() + static_cast<std::size_t>(ncd) * (ng + 1); }
  std::size_t s() const { return hh() + static_cast<std::size_t>(ncd) * (ng + 1); }
  std::size_t chi() const { return s() + static_cast<std::size_t>(ncd) * (ng + 1); }
  std::size_t m0() const { return chi() + (nc + 1); }
  std::size_t size() const { return m0() + (nc + 1); }

  // Perturbative order of coefficient i.
  int order(std::size_t i) const {
    if (i < hp()) return static_cast<int>(i);
    if (i < chi()) return static_cast<int>((i - hp()) % (ng + 1));
    return static_cast<int>(i < m0() ? i - chi() : i - m0());
  }
};

struct PassOptions {
  int threads = 1;
  bool verbose = true;
  bool cap_largest = true;  // largest classes in the occupation-capped model (exact; see pipeline.cpp)
};

// Returns, per lane, the lattice series residues in SeriesLayout order.
std::vector<std::vector<u64>> run_pass(const Geometry& geo, int ng, int nc, const std::vector<LaneSpec>& lanes,
                                       const PassOptions& opts);

}  // namespace nlce
