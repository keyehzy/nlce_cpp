#include "geometry.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace nlce {

namespace {

std::string key_string(int cls, const Pattern& pattern) {
  std::string s(sizeof(int) + pattern.size(), '\0');
  std::copy_n(reinterpret_cast<const char*>(&cls), sizeof(int), s.data());
  std::copy(pattern.begin(), pattern.end(), s.data() + sizeof(int));
  return s;
}

ClassInfo make_class(const Cert& cert) {
  ClassInfo cls;
  cls.cert = cert;
  cls.nv = cert.n;
  cls.edges = edges_from_cert(cert);
  cls.edge_index.assign(cls.nv * cls.nv, -1);
  for (int e = 0; e < static_cast<int>(cls.edges.size()); ++e) {
    auto [i, j] = cls.edges[e];
    cls.edge_index[i * cls.nv + j] = e;
    cls.edge_index[j * cls.nv + i] = e;
  }
  cls.autos = automorphisms(cls.nv, cls.edges);
  cls.matching = double_cover_matching(cls.nv, cls.edges);
  return cls;
}

}  // namespace

int Geometry::key_index(int cls, const Pattern& pattern) const {
  auto it = key_of.find(key_string(cls, pattern));
  if (it == key_of.end()) throw std::logic_error("decorated subcluster missing from geometry");
  return it->second;
}

Pattern canonical_pattern(const ClassInfo& cls, const Pattern& signs) {
  Pattern best;
  Pattern image(signs.size());
  for (const auto& perm : cls.autos) {
    for (std::size_t e = 0; e < cls.edges.size(); ++e) {
      auto [i, j] = cls.edges[e];
      const int a = perm[i], b = perm[j];
      image[cls.edge_index[a * cls.nv + b]] = static_cast<std::int8_t>(a < b ? signs[e] : -signs[e]);
    }
    for (int flip : {1, -1}) {
      Pattern cand = image;
      if (flip < 0) {
        for (auto& x : cand) x = static_cast<std::int8_t>(-x);
      }
      if (best.empty() && !cand.empty()) best = cand;
      if (cand < best) best = cand;
    }
  }
  return best;
}

Geometry build_geometry(const Lattice& lat, int smax) {
  Geometry geo;
  geo.lattice = &lat;
  geo.smax = smax;
  geo.cluster_counts.assign(smax + 1, 0);
  const auto reps = enumerate_site_clusters(lat, smax);
  const std::int64_t order = static_cast<std::int64_t>(lat.group.size());
  std::map<Site, int> disp_index;
  std::map<std::pair<Site, Site>, int> pair_index;
  const bool resolve_pairs = lat.cell_sites() > 1;
  std::vector<std::map<std::tuple<int, int, int>, std::int64_t>> emb(0), pair_emb(0);

  for (int s = 1; s <= smax; ++s) {
    geo.cluster_counts[s] = static_cast<int>(reps[s].size());
    for (const auto& sites : reps[s]) {
      const int n = static_cast<int>(sites.size());
      const auto bonds = induced_bonds(lat, sites);
      std::vector<Edge> edges;
      edges.reserve(bonds.size());
      auto pos = [&](Site x) {
        return static_cast<int>(std::lower_bound(sites.begin(), sites.end(), x) - sites.begin());
      };
      for (const auto& [p, q] : bonds) edges.emplace_back(pos(p), pos(q));
      const Canon canon = canonical_form(n, edges);

      auto [it, inserted] = geo.class_of.try_emplace(canon.cert, static_cast<int>(geo.classes.size()));
      if (inserted) {
        geo.classes.push_back(make_class(canon.cert));
        emb.emplace_back();
        pair_emb.emplace_back();
      }
      const int c = it->second;
      ClassInfo& cls = geo.classes[c];
      const int orbit = site_orbit_size(lat, sites);
      cls.mult += orbit;
      Realization real{orbit, std::vector<Site>(n)};
      for (int u = 0; u < n; ++u) real.pos[canon.perm[u]] = sites[u];
      cls.realizations.push_back(std::move(real));

      for (int u = 0; u < n; ++u) {
        for (int w = 0; w < n; ++w) {
          const Site d = sites[w] - sites[u];
          const Site cd = canonical_displacement(lat, d);
          auto [dit, fresh] = disp_index.try_emplace(cd, static_cast<int>(geo.displacements.size()));
          if (fresh) geo.displacements.push_back(cd);
          emb[c][{dit->second, canon.perm[u], canon.perm[w]}] +=
              static_cast<std::int64_t>(orbit) * order / displacement_orbit_size(lat, d);
          if (!resolve_pairs) continue;
          const auto cp = canonical_pair(lat, sites[u], sites[w]);
          auto [pit, fresh_pair] = pair_index.try_emplace(cp, static_cast<int>(geo.pairs.size()));
          if (fresh_pair) {
            geo.pairs.push_back({cp.first, cp.second});
            geo.pair_counts.push_back(pair_orbit_size(lat, sites[u], sites[w]));
          }
          pair_emb[c][{pit->second, canon.perm[u], canon.perm[w]}] += orbit;
        }
      }
      if (!lat.currents()) continue;

      Pattern signs(cls.edges.size());
      for (const auto& [p, q] : bonds) {
        int i = canon.perm[pos(p)], j = canon.perm[pos(q)];
        int sgn = current_sign(lat, p, q);
        if (i > j) {
          std::swap(i, j);
          sgn = -sgn;
        }
        signs[cls.edge_index[i * cls.nv + j]] = static_cast<std::int8_t>(sgn);
      }
      const Pattern pattern = canonical_pattern(cls, signs);
      auto [kit, new_key] = geo.key_of.try_emplace(key_string(c, pattern), static_cast<int>(geo.keys.size()));
      if (new_key) {
        geo.keys.push_back({c, pattern, 0});
        cls.keys.push_back(kit->second);
      }
      geo.keys[kit->second].mult += orbit;
    }
  }

  for (std::size_t c = 0; c < geo.classes.size(); ++c) {
    for (const auto& [k, fac] : emb[c]) {
      auto [cd, a, b] = k;
      geo.classes[c].embeddings.push_back({cd, a, b, fac});
    }
    for (const auto& [k, fac] : pair_emb[c]) {
      auto [pc, a, b] = k;
      geo.classes[c].pair_embeddings.push_back({pc, a, b, fac});
    }
  }
  return geo;
}

namespace {

// (anchor of the holon's sublattice, doublon - holon) of the pair (i, j).
std::pair<Site, Site> pair_state(const Lattice& lat, Site i, Site j) {
  return {*lat.anchors[lat.residue(j)], i - j};
}

std::array<int, 8> dh_flat(std::pair<Site, Site> after, std::pair<Site, Site> before) {
  return {after.first.a,  after.first.b,  after.second.a,  after.second.b,
          before.first.a, before.first.b, before.second.a, before.second.b};
}

}  // namespace

int Geometry::dh_key(Site i2, Site j2, Site i, Site j) const {
  auto it = dh_key_of.find(dh_flat(pair_state(*lattice, i2, j2), pair_state(*lattice, i, j)));
  if (it == dh_key_of.end()) throw std::logic_error("doublon-holon key missing from geometry");
  return it->second;
}

void build_dh_keys(Geometry& geo) {
  const Lattice& lat = *geo.lattice;
  std::set<std::pair<Site, Site>> states;
  for (const auto& cls : geo.classes) {
    for (const auto& real : cls.realizations) {
      for (const auto& p : real.pos) {
        for (const auto& q : real.pos) {
          if (!(p == q)) states.insert(pair_state(lat, q, p));
        }
      }
    }
  }
  geo.dh_keys.clear();
  geo.dh_key_orbit.clear();
  geo.dh_key_of.clear();
  for (const auto& s2 : states) {
    for (const auto& s : states) {
      if (geo.dh_key_of.count(dh_flat(s2, s))) continue;
      std::set<std::array<int, 8>> images;
      for (const auto& g : lat.group) {
        const Site h2 = g(s2.first), h = g(s.first);
        images.insert(dh_flat(pair_state(lat, h2 + g(s2.second), h2), pair_state(lat, h + g(s.second), h)));
      }
      const auto& c = *images.begin();
      const int key = static_cast<int>(geo.dh_keys.size());
      geo.dh_keys.push_back({Site{c[0], c[1]}, Site{c[2], c[3]}, Site{c[4], c[5]}, Site{c[6], c[7]}});
      geo.dh_key_orbit.push_back(static_cast<int>(images.size()));
      for (const auto& im : images) geo.dh_key_of.emplace(im, key);
    }
  }
}

std::vector<SubCluster> subclusters(const Geometry& geo, int cls_index) {
  const ClassInfo& cls = geo.classes[cls_index];
  const int n = cls.nv;
  std::vector<std::uint32_t> adj(n, 0);
  for (auto [i, j] : cls.edges) {
    adj[i] |= 1u << j;
    adj[j] |= 1u << i;
  }
  std::vector<SubCluster> out;
  for (std::uint32_t mask = 1; mask + 1 < (1u << n); ++mask) {
    const int root = __builtin_ctz(mask);
    std::uint32_t seen = 1u << root, frontier = seen;
    while (frontier) {
      std::uint32_t next = 0;
      for (std::uint32_t m = frontier; m; m &= m - 1) next |= adj[__builtin_ctz(m)];
      next &= mask & ~seen;
      seen |= next;
      frontier = next;
    }
    if (seen != mask) continue;

    SubCluster sub;
    std::vector<int> local(n, -1);
    for (int v = 0; v < n; ++v) {
      if (mask >> v & 1u) {
        local[v] = static_cast<int>(sub.verts.size());
        sub.verts.push_back(v);
      }
    }
    std::vector<Edge> edges;
    for (int e = 0; e < static_cast<int>(cls.edges.size()); ++e) {
      auto [i, j] = cls.edges[e];
      if (local[i] >= 0 && local[j] >= 0) {
        edges.emplace_back(local[i], local[j]);
        sub.parent_edges.push_back(e);
      }
    }
    const Canon canon = canonical_form(static_cast<int>(sub.verts.size()), edges);
    auto it = geo.class_of.find(canon.cert);
    if (it == geo.class_of.end()) throw std::logic_error("subcluster class missing from geometry");
    sub.cls = it->second;
    sub.map = canon.perm;
    out.push_back(std::move(sub));
  }
  return out;
}

int subcluster_key(const Geometry& geo, int parent_cls, const SubCluster& sub, const Pattern& parent) {
  const ClassInfo& pc = geo.classes[parent_cls];
  const ClassInfo& sc = geo.classes[sub.cls];
  std::vector<int> local(pc.nv, -1);
  for (std::size_t k = 0; k < sub.verts.size(); ++k) local[sub.verts[k]] = static_cast<int>(k);
  Pattern signs(sc.edges.size());
  for (int e : sub.parent_edges) {
    auto [p, q] = pc.edges[e];
    int i = sub.map[local[p]], j = sub.map[local[q]];
    int sgn = parent[e];
    if (i > j) {
      std::swap(i, j);
      sgn = -sgn;
    }
    signs[sc.edge_index[i * sc.nv + j]] = static_cast<std::int8_t>(sgn);
  }
  return geo.key_index(sub.cls, canonical_pattern(sc, signs));
}

}  // namespace nlce
