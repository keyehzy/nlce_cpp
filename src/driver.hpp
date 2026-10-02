// Multi-modular drivers: choose primes, run the modular computation, and
// reconstruct exact rationals, adding primes until the reconstruction is
// confirmed by `check_primes` further primes.
#pragma once

#include "cluster.hpp"
#include "geometry.hpp"
#include "rational.hpp"
#include "reconstruct.hpp"

#include <functional>
#include <vector>

namespace nlce {

struct DriverOptions {
  int threads = 1;
  int fixed_primes = 0;  // > 0: use exactly this many and fail if too few
  int check_primes = 2;
  bool verbose = true;
};

struct LatticeSeries {
  Rational v;
  int primes = 0;      // reconstruction primes, check primes excluded
  Reconstruction rec;  // values in SeriesLayout order
};

// Lattice series for every V/U in `vs`.  All V/U share each pass, so `done`
// is called for each one as soon as its series is reconstructed.
void lattice_series(const Geometry& geo, int ng, int nc, const std::vector<Rational>& vs, const DriverOptions& opts,
                    const std::function<void(const LatticeSeries&)>& done);

struct ClusterSeries {
  int primes = 0;      // reconstruction primes, check primes excluded
  Reconstruction rec;  // values in RawSeries order: E, Hp, Hh, corr, chi, m0
};

// Finite-cluster series of one cluster for one V/U.
ClusterSeries cluster_series(const ClusterPlan& plan, const Rational& v, int check_primes);

}  // namespace nlce
