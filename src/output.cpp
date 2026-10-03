#include "output.hpp"

#include "pipeline.hpp"

#include <string>
#include <utility>

namespace nlce {

namespace {

void json_list(std::ostream& os, const std::vector<std::string>& v, std::size_t begin, std::size_t n) {
  os << '[';
  for (std::size_t i = 0; i < n; ++i) os << (i ? "," : "") << '"' << v[begin + i] << '"';
  os << ']';
}

}  // namespace

void write_lattice_json(std::ostream& os, const Geometry& geo, int nsites, int ng, int nc, int ndh,
                        const LatticeSeries& series, int check_primes) {
  const Lattice& lattice = *geo.lattice;
  const std::vector<Site>& displacements = geo.displacements;
  const std::vector<std::array<Site, 4>>& dh_keys = geo.dh_keys;
  const Reconstruction& rec = series.rec;
  const SeriesLayout lay{ng,
                         nc,
                         static_cast<int>(displacements.size()),
                         ndh,
                         static_cast<int>(dh_keys.size()),
                         static_cast<int>(geo.pairs.size())};
  os << "{\n\"lattice\": \"" << lattice.name << "\",\n\"nsites\": " << nsites << ",\n\"v\": \""
     << to_string(series.v) << "\",\n\"order_gap\": " << ng;
  if (lattice.currents()) {
    os << ",\n\"current\": \"" << (lattice.current == Current::uniform ? "uniform" : "staggered")
       << "\",\n\"order_chi\": " << nc;
  }
  os << ",\n\"primes\": " << series.primes << ",\n\"check_primes\": " << check_primes
     << ",\n\"max_bits\": " << rec.max_bits << ",\n\"displacements\": [";
  for (std::size_t d = 0; d < displacements.size(); ++d) {
    os << (d ? "," : "") << '[' << displacements[d].a << ',' << displacements[d].b << ']';
  }
  os << "],\n\"orbit_sizes\": [";
  for (std::size_t d = 0; d < displacements.size(); ++d) {
    os << (d ? "," : "") << displacement_orbit_size(lattice, displacements[d]);
  }
  os << "],\n\"EN\": ";
  json_list(os, rec.values, lay.en(), ng + 1);
  for (auto [name, off] : {std::pair{"Hp", lay.hp()}, std::pair{"Hh", lay.hh()}, std::pair{"S", lay.s()}}) {
    os << ",\n\"" << name << "\": [";
    for (int d = 0; d < lay.ncd; ++d) {
      os << (d ? ",\n  " : "");
      json_list(os, rec.values, off + static_cast<std::size_t>(d) * (ng + 1), ng + 1);
    }
    os << ']';
  }
  if (lattice.currents()) {
    os << ",\n\"chi\": ";
    json_list(os, rec.values, lay.chi(), nc + 1);
    os << ",\n\"m0\": ";
    json_list(os, rec.values, lay.m0(), nc + 1);
  }
  if (ndh >= 0) {
    os << ",\n\"order_dh\": " << ndh << ",\n\"dh_keys\": [";
    for (std::size_t q = 0; q < dh_keys.size(); ++q) {
      const auto& [h2, r2, h, r] = dh_keys[q];
      os << (q ? "," : "") << '[' << r2.a << ',' << r2.b << ',' << r.a << ',' << r.b << ']';
    }
    if (lattice.cell_sites() > 1) {
      os << "],\n\"dh_holons\": [";
      for (std::size_t q = 0; q < dh_keys.size(); ++q) {
        const auto& [h2, r2, h, r] = dh_keys[q];
        os << (q ? "," : "") << '[' << h2.a << ',' << h2.b << ',' << h.a << ',' << h.b << ']';
      }
    }
    os << "],\n\"Idh\": [";
    for (std::size_t q = 0; q < dh_keys.size(); ++q) {
      os << (q ? ",\n  " : "");
      json_list(os, rec.values, lay.dh() + q * (ndh + 1), ndh + 1);
    }
    os << ']';
  }
  if (!geo.pairs.empty()) {
    os << ",\n\"pairs\": [";
    for (std::size_t q = 0; q < geo.pairs.size(); ++q) {
      const auto& [p, r] = geo.pairs[q];
      os << (q ? "," : "") << '[' << p.a << ',' << p.b << ',' << r.a << ',' << r.b << ']';
    }
    os << "],\n\"pair_counts\": [";
    for (std::size_t q = 0; q < geo.pairs.size(); ++q) os << (q ? "," : "") << geo.pair_counts[q];
    os << ']';
    for (auto [name, off] : {std::pair{"Hp_pairs", lay.hp_pairs()}, std::pair{"Hh_pairs", lay.hh_pairs()}}) {
      os << ",\n\"" << name << "\": [";
      for (std::size_t q = 0; q < geo.pairs.size(); ++q) {
        os << (q ? ",\n  " : "");
        json_list(os, rec.values, off + q * (ng + 1), ng + 1);
      }
      os << ']';
    }
  }
  os << "\n}\n";
}

void write_cluster_json(std::ostream& os, int nv, int ng, int nc, int npatterns, int ndh,
                        const ClusterSeries& series) {
  const std::vector<std::string>& values = series.rec.values;
  std::size_t at = 0;
  auto emit = [&](const char* name, int blocks, int len) {
    os << (at ? ",\n" : "{\n") << '"' << name << "\": [";
    for (int q = 0; q < blocks; ++q) {
      os << (q ? "," : "");
      json_list(os, values, at, len);
      at += len;
    }
    os << ']';
  };
  emit("E", 1, ng + 1);
  emit("Hp", nv * nv, ng + 1);
  emit("Hh", nv * nv, ng + 1);
  emit("corr", nv * nv, ng + 1);
  emit("chi", npatterns, nc + 1);
  emit("m0", npatterns, nc + 1);
  if (ndh >= 0) emit("Idh", nv * nv * nv * nv, ndh + 1);
  os << ",\n\"primes\": " << series.primes << "\n}\n";
}

}  // namespace nlce
