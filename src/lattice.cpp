#include "lattice.hpp"

#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include <algorithm>
#include <set>
#include <stdexcept>

namespace nlce {

namespace {

PointOp compose(const PointOp& p, const PointOp& q) {
  return {p.m0 * q.m0 + p.m1 * q.m2, p.m0 * q.m1 + p.m1 * q.m3,
          p.m2 * q.m0 + p.m3 * q.m2, p.m2 * q.m1 + p.m3 * q.m3};
}

// Closure of `generators` under composition, identity first.
std::vector<PointOp> generate_group(const std::vector<PointOp>& generators, std::size_t order) {
  std::vector<PointOp> group{{1, 0, 0, 1}};
  std::vector<PointOp> frontier = group;
  while (!frontier.empty()) {
    std::vector<PointOp> next;
    for (const auto& g : frontier) {
      for (const auto& h : generators) {
        PointOp x = compose(g, h);
        if (std::find(group.begin(), group.end(), x) == group.end()) {
          group.push_back(x);
          next.push_back(x);
        }
      }
    }
    frontier = std::move(next);
  }
  if (group.size() != order) throw std::logic_error("point group has the wrong order");
  return group;
}

struct SitesHash {
  std::size_t operator()(const std::vector<Site>& v) const {
    std::size_t h = v.size();
    for (const auto& s : v) {
      boost::hash_combine(h, s.a);
      boost::hash_combine(h, s.b);
    }
    return h;
  }
};

std::vector<Site> normalised_image(const Lattice& lat, const PointOp& m, const std::vector<Site>& sites) {
  std::vector<Site> img;
  img.reserve(sites.size());
  for (const auto& s : sites) img.push_back(m(s));
  const Site o = *std::min_element(img.begin(), img.end());
  const Site shift = *lat.anchors[lat.residue(o)] - o;
  for (auto& s : img) s = s + shift;
  std::sort(img.begin(), img.end());
  return img;
}

// 60-degree rotation (a, b) -> (-b, a + b) and the mirror about e1.
const std::vector<PointOp>& d6_group() {
  static const std::vector<PointOp> group = generate_group({{0, -1, 1, 1}, {1, 1, 0, -1}}, 12);
  return group;
}

}  // namespace

int Lattice::residue(Site s) const {
  // s minus k (q, r) lies in the first row of cells, 0 <= b < r.
  const int k = s.b >= 0 ? s.b / cell.r : -((-s.b + cell.r - 1) / cell.r);
  const int a = ((s.a - k * cell.q) % cell.p + cell.p) % cell.p;
  return (s.b - k * cell.r) * cell.p + a;
}

int Lattice::cell_sites() const {
  return static_cast<int>(std::count_if(anchors.begin(), anchors.end(), [](const auto& a) { return a.has_value(); }));
}

const Lattice& chain_lattice() {
  // Inversion (a, 0) -> (-a, 0).
  static const Lattice lat{.name = "chain", .dirs = {{1, 0}}, .group = generate_group({{-1, 0, 0, -1}}, 2)};
  return lat;
}

const Lattice& square_lattice() {
  // 90-degree rotation (a, b) -> (-b, a) and the mirror about e1.
  static const Lattice lat{.name = "square",
                           .dirs = {{1, 0}, {0, 1}},
                           .group = generate_group({{0, -1, 1, 0}, {1, 0, 0, -1}}, 8)};
  return lat;
}

const Lattice& triangular_lattice() {
  static const Lattice lat{.name = "triangular", .dirs = {{1, 0}, {0, 1}, {1, -1}}, .group = d6_group(),
                           .currents = true};
  return lat;
}

const Lattice& honeycomb_lattice() {
  // Translations (3, 0) and (1, 1), so the residue is (a - b) mod 3: 0 for the
  // hexagon centres, 1 and 2 for the two sublattices.  Every triangular bond
  // joins different residues, so between sites they are exactly the
  // honeycomb bonds.  The D6 about a hexagon centre maps residue r to -r and
  // so preserves the sites and the translations.
  static const Lattice lat{.name = "honeycomb",
                           .dirs = {{1, 0}, {0, 1}, {1, -1}},
                           .group = d6_group(),
                           .cell = {3, 1, 1},
                           .anchors = {std::nullopt, Site{1, 0}, Site{0, 1}}};
  return lat;
}

const Lattice& kagome_lattice() {
  // Translations (2, 0) and (0, 2); the residue (a mod 2) + 2 (b mod 2) is 0
  // for the hexagon centres and 1, 2, 3 for the three sublattices.  Each site
  // keeps four of its six triangular neighbours, the other two being hexagon
  // centres, and the kept bonds are those of the kagome lattice.  The D6
  // about a hexagon centre preserves the even points, hence the sites and
  // the translations.
  static const Lattice lat{.name = "kagome",
                           .dirs = {{1, 0}, {0, 1}, {1, -1}},
                           .group = d6_group(),
                           .cell = {2, 0, 2},
                           .anchors = {std::nullopt, Site{1, 0}, Site{0, 1}, Site{1, 1}}};
  return lat;
}

const Lattice& lattice_by_name(const std::string& name) {
  for (const Lattice* lat :
       {&chain_lattice(), &square_lattice(), &triangular_lattice(), &honeycomb_lattice(), &kagome_lattice()}) {
    if (name == lat->name) return *lat;
  }
  throw std::invalid_argument("unknown lattice " + name + " (chain, square, triangular, honeycomb, kagome)");
}

Site canonical_displacement(const Lattice& lat, Site d) {
  Site best = d;
  for (const auto& m : lat.group) best = std::min(best, m(d));
  return best;
}

int displacement_orbit_size(const Lattice& lat, Site d) {
  std::set<Site> images;
  for (const auto& m : lat.group) images.insert(m(d));
  return static_cast<int>(images.size());
}

int sublattice(Site s) { return ((2 * s.a + s.b) % 3 + 3) % 3; }

int current_sign(Site p, Site q) { return ((sublattice(q) - sublattice(p)) % 3 + 3) % 3 == 1 ? 1 : -1; }

std::vector<Bond> induced_bonds(const Lattice& lat, const std::vector<Site>& sites) {
  std::set<Site> members(sites.begin(), sites.end());
  std::vector<Bond> out;
  for (const auto& s : sites) {
    for (const auto& d : lat.dirs) {
      Site t = s + d;
      if (members.count(t)) out.push_back(s <= t ? Bond{s, t} : Bond{t, s});
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<Site> canonical_sites(const Lattice& lat, const std::vector<Site>& sites) {
  std::vector<Site> best;
  for (const auto& m : lat.group) {
    auto img = normalised_image(lat, m, sites);
    if (best.empty() || img < best) best = std::move(img);
  }
  return best;
}

int site_orbit_size(const Lattice& lat, const std::vector<Site>& sites) {
  std::set<std::vector<Site>> forms;
  for (const auto& m : lat.group) forms.insert(normalised_image(lat, m, sites));
  return static_cast<int>(forms.size());
}

std::vector<std::vector<std::vector<Site>>> enumerate_site_clusters(const Lattice& lat, int smax) {
  if (smax < 1) throw std::invalid_argument("smax must be positive");
  std::vector<std::vector<std::vector<Site>>> reps(smax + 1);
  for (const auto& anchor : lat.anchors) {
    if (!anchor) continue;
    auto key = canonical_sites(lat, {*anchor});
    if (std::find(reps[1].begin(), reps[1].end(), key) == reps[1].end()) reps[1].push_back(std::move(key));
  }
  for (int s = 1; s < smax; ++s) {
    boost::unordered_flat_set<std::vector<Site>, SitesHash> seen;
    auto& next = reps[s + 1];
    for (const auto& cluster : reps[s]) {
      std::set<Site> members(cluster.begin(), cluster.end());
      std::set<Site> candidates;
      for (const auto& u : cluster) {
        for (const auto& d : lat.dirs) {
          for (Site w : {u + d, u - d}) {
            if (lat.is_site(w) && !members.count(w)) candidates.insert(w);
          }
        }
      }
      for (const auto& w : candidates) {
        auto grown = cluster;
        grown.push_back(w);
        auto key = canonical_sites(lat, grown);
        if (seen.insert(key).second) next.push_back(std::move(key));
      }
    }
  }
  return reps;
}

}  // namespace nlce
