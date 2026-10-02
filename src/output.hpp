// JSON output of nlce_run, every coefficient an exact rational string.
#pragma once

#include "driver.hpp"
#include "lattice.hpp"

#include <ostream>
#include <vector>

namespace nlce {

// Lattice series of one V/U, read by tools/to_pickles.py.
void write_lattice_json(std::ostream& os, int nsites, int ng, int nc, const std::vector<Site>& displacements,
                        const LatticeSeries& series, int check_primes);

// Finite-cluster series, read by tools/check_cluster.py: E, then Hp, Hh and
// corr per vertex pair (u*nv + w), then chi and m0 per pattern.
void write_cluster_json(std::ostream& os, int nv, int ng, int nc, int npatterns, const ClusterSeries& series);

}  // namespace nlce
