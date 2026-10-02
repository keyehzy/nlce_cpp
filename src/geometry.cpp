#include "geometry.hpp"

#include <algorithm>
#include <map>
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

Geometry build_geometry(int smax) {
  Geometry geo;
  geo.smax = smax;
  geo.cluster_counts.assign(smax + 1, 0);
  const auto reps = enumerate_site_clusters(smax);
  std::map<Site, int> disp_index;
  std::vector<std::map<std::tuple<int, int, int>, std::int64_t>> emb(0);

  for (int s = 1; s <= smax; ++s) {
    geo.cluster_counts[s] = static_cast<int>(reps[s].size());
    for (const auto& sites : reps[s]) {
      const int n = static_cast<int>(sites.size());
      const auto bonds = induced_bonds(sites);
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
      }
      const int c = it->second;
      ClassInfo& cls = geo.classes[c];
      const int orbit = site_orbit_size(sites);
      cls.mult += orbit;

      for (int u = 0; u < n; ++u) {
        for (int w = 0; w < n; ++w) {
          const Site d = sites[w] - sites[u];
          const Site cd = canonical_displacement(d);
          auto [dit, fresh] = disp_index.try_emplace(cd, static_cast<int>(geo.displacements.size()));
          if (fresh) geo.displacements.push_back(cd);
          emb[c][{dit->second, canon.perm[u], canon.perm[w]}] +=
              static_cast<std::int64_t>(orbit) * 12 / displacement_orbit_size(d);
        }
      }

      Pattern signs(cls.edges.size());
      for (const auto& [p, q] : bonds) {
        int i = canon.perm[pos(p)], j = canon.perm[pos(q)];
        int sgn = current_sign(p, q);
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
    for (const auto& [k, fac12] : emb[c]) {
      auto [cd, a, b] = k;
      geo.classes[c].embeddings.push_back({cd, a, b, fac12});
    }
  }
  return geo;
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
