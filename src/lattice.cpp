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

bool same(const PointOp& p, const PointOp& q) {
  return p.m0 == q.m0 && p.m1 == q.m1 && p.m2 == q.m2 && p.m3 == q.m3;
}

std::array<PointOp, 12> make_point_group() {
  // 60-degree rotation (a, b) -> (-b, a + b) and the mirror about e1.
  const PointOp r6{0, -1, 1, 1};
  const PointOp mirror{1, 1, 0, -1};
  std::vector<PointOp> group{{1, 0, 0, 1}};
  std::vector<PointOp> frontier = group;
  while (!frontier.empty()) {
    std::vector<PointOp> next;
    for (const auto& g : frontier) {
      for (const auto& h : {r6, mirror}) {
        PointOp x = compose(g, h);
        bool known = std::any_of(group.begin(), group.end(), [&](const PointOp& y) { return same(x, y); });
        if (!known) {
          group.push_back(x);
          next.push_back(x);
        }
      }
    }
    frontier = std::move(next);
  }
  if (group.size() != 12) throw std::logic_error("triangular point group must have 12 elements");
  std::array<PointOp, 12> out{};
  std::copy(group.begin(), group.end(), out.begin());
  return out;
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

std::vector<Site> normalised_image(const PointOp& m, const std::vector<Site>& sites) {
  std::vector<Site> img;
  img.reserve(sites.size());
  for (const auto& s : sites) img.push_back(m(s));
  Site o = *std::min_element(img.begin(), img.end());
  for (auto& s : img) s = s - o;
  std::sort(img.begin(), img.end());
  return img;
}

}  // namespace

const std::array<PointOp, 12>& point_group() {
  static const std::array<PointOp, 12> group = make_point_group();
  return group;
}

Site canonical_displacement(Site d) {
  Site best = point_group()[0](d);
  for (const auto& m : point_group()) best = std::min(best, m(d));
  return best;
}

int displacement_orbit_size(Site d) {
  std::set<Site> images;
  for (const auto& m : point_group()) images.insert(m(d));
  return static_cast<int>(images.size());
}

int sublattice(Site s) { return ((2 * s.a + s.b) % 3 + 3) % 3; }

int current_sign(Site p, Site q) { return ((sublattice(q) - sublattice(p)) % 3 + 3) % 3 == 1 ? 1 : -1; }

std::vector<Bond> induced_bonds(const std::vector<Site>& sites) {
  std::set<Site> members(sites.begin(), sites.end());
  std::vector<Bond> out;
  for (const auto& s : sites) {
    for (const auto& d : kDirs) {
      Site t = s + d;
      if (members.count(t)) out.push_back(s <= t ? Bond{s, t} : Bond{t, s});
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<Site> canonical_sites(const std::vector<Site>& sites) {
  std::vector<Site> best;
  for (const auto& m : point_group()) {
    auto img = normalised_image(m, sites);
    if (best.empty() || img < best) best = std::move(img);
  }
  return best;
}

int site_orbit_size(const std::vector<Site>& sites) {
  std::set<std::vector<Site>> forms;
  for (const auto& m : point_group()) forms.insert(normalised_image(m, sites));
  return static_cast<int>(forms.size());
}

std::vector<std::vector<std::vector<Site>>> enumerate_site_clusters(int smax) {
  if (smax < 1) throw std::invalid_argument("smax must be positive");
  std::vector<std::vector<std::vector<Site>>> reps(smax + 1);
  reps[1].push_back(canonical_sites({{0, 0}}));
  for (int s = 1; s < smax; ++s) {
    boost::unordered_flat_set<std::vector<Site>, SitesHash> seen;
    auto& next = reps[s + 1];
    for (const auto& cluster : reps[s]) {
      std::set<Site> members(cluster.begin(), cluster.end());
      std::set<Site> candidates;
      for (const auto& u : cluster) {
        for (const auto& d : kDirs) {
          for (Site w : {u + d, u - d}) {
            if (!members.count(w)) candidates.insert(w);
          }
        }
      }
      for (const auto& w : candidates) {
        auto grown = cluster;
        grown.push_back(w);
        auto key = canonical_sites(grown);
        if (seen.insert(key).second) next.push_back(std::move(key));
      }
    }
  }
  return reps;
}

}  // namespace nlce
