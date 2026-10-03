// JSON output of nlce_run, every coefficient an exact rational string.
#pragma once

#include "driver.hpp"
#include "lattice.hpp"

#include <ostream>
#include <vector>

namespace nlce {

// Lattice series of one V/U per site, read by tools/to_pickles.py.  Hp, Hh
// and S are given per canonical displacement, averaged over its point-group
// orbit (whose size is listed alongside) and over the starting site in a unit
// cell.  On a Bravais lattice every displacement of the orbit has that value;
// on the honeycomb lattice the averages still give the q = 0 sums exactly.
// chi and m0 only on lattices with the staggered current.
void write_lattice_json(std::ostream& os, const Lattice& lattice, int nsites, int ng, int nc,
                        const std::vector<Site>& displacements, const LatticeSeries& series, int check_primes);

// Finite-cluster series, read by tools/check_cluster.py: E, then Hp, Hh and
// corr per vertex pair (u*nv + w), then chi and m0 per pattern.
void write_cluster_json(std::ostream& os, int nv, int ng, int nc, int npatterns, const ClusterSeries& series);

}  // namespace nlce
