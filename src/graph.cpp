#include "graph.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <stdexcept>

namespace nlce {

namespace {

using Colors = std::array<int, kMaxVertices>;
using Adjacency = std::array<std::uint32_t, kMaxVertices>;

int pair_index(int n, int i, int j) { return i * n - i * (i + 1) / 2 + (j - i - 1); }

u128 pair_bit(int n, int i, int j) { return static_cast<u128>(1) << (127 - pair_index(n, i, j)); }

Adjacency adjacency(int n, const std::vector<Edge>& edges) {
  if (n > kMaxVertices) throw std::invalid_argument("graph too large for canonical labelling");
  Adjacency adj{};
  for (auto [u, v] : edges) {
    if (u == v || u < 0 || v < 0 || u >= n || v >= n) throw std::invalid_argument("bad edge");
    adj[u] |= 1u << v;
    adj[v] |= 1u << u;
  }
  return adj;
}

// Equitable refinement: a vertex's new colour is the rank of
// (colour, degree, sorted neighbour colours) among all vertices.
void refine(int n, const Adjacency& adj, Colors& colors) {
  struct Sig {
    std::array<int, kMaxVertices + 2> key;
    int len;
    int vertex;
  };
  std::array<Sig, kMaxVertices> sig{};
  while (true) {
    for (int i = 0; i < n; ++i) {
      Sig& s = sig[i];
      s.vertex = i;
      s.key[0] = colors[i];
      int deg = 0;
      std::array<int, kMaxVertices> nb{};
      for (std::uint32_t m = adj[i]; m; m &= m - 1) nb[deg++] = colors[__builtin_ctz(m)];
      std::sort(nb.begin(), nb.begin() + deg);
      s.key[1] = deg;
      std::copy(nb.begin(), nb.begin() + deg, s.key.begin() + 2);
      s.len = deg + 2;
    }
    auto less = [](const Sig& x, const Sig& y) {
      return std::lexicographical_compare(x.key.begin(), x.key.begin() + x.len, y.key.begin(),
                                          y.key.begin() + y.len);
    };
    std::array<Sig, kMaxVertices> sorted = sig;
    std::sort(sorted.begin(), sorted.begin() + n, less);
    Colors next{};
    int rank = 0;
    for (int k = 0; k < n; ++k) {
      if (k > 0 && less(sorted[k - 1], sorted[k])) ++rank;
      next[sorted[k].vertex] = rank;
    }
    bool stable = true;
    for (int i = 0; i < n; ++i) stable = stable && next[i] == colors[i];
    colors = next;
    if (stable) return;
  }
}

struct Search {
  int n;
  Adjacency adj;
  bool have = false;
  Cert best;
  std::vector<int> best_perm;

  void leaf(const Colors& colors) {
    u128 bits = 0;
    for (int i = 0; i < n; ++i) {
      for (std::uint32_t m = adj[i]; m; m &= m - 1) {
        int j = __builtin_ctz(m);
        int a = colors[i], b = colors[j];
        if (a < b) bits |= pair_bit(n, a, b);
      }
    }
    if (!have || bits < best.bits) {
      have = true;
      best = {n, bits};
      best_perm.assign(colors.begin(), colors.begin() + n);
    }
  }

  void recurse(Colors colors) {
    refine(n, adj, colors);
    std::array<int, kMaxVertices> count{};
    for (int i = 0; i < n; ++i) ++count[colors[i]];
    int target = -1;
    for (int c = 0; c < n; ++c) {
      if (count[c] > 1) {
        target = c;
        break;
      }
    }
    if (target < 0) {
      leaf(colors);
      return;
    }
    for (int v = 0; v < n; ++v) {
      if (colors[v] != target) continue;
      Colors next = colors;
      for (int i = 0; i < n; ++i) {
        if (colors[i] > target || (colors[i] == target && i != v)) ++next[i];
      }
      recurse(next);
    }
  }
};

}  // namespace

std::size_t CertHash::operator()(const Cert& c) const {
  u64 lo = static_cast<u64>(c.bits), hi = static_cast<u64>(c.bits >> 64);
  std::size_t h = std::hash<u64>{}(lo ^ (hi * 0x9e3779b97f4a7c15ULL));
  return h ^ (static_cast<std::size_t>(c.n) * 0xc2b2ae3d27d4eb4fULL);
}

Canon canonical_form(int n, const std::vector<Edge>& edges) {
  if (n == 0) return {{0, 0}, {}};
  Search search{n, adjacency(n, edges), false, {}, {}};
  search.recurse(Colors{});
  return {search.best, search.best_perm};
}

std::vector<Edge> edges_from_cert(const Cert& cert) {
  std::vector<Edge> out;
  for (int i = 0; i < cert.n; ++i) {
    for (int j = i + 1; j < cert.n; ++j) {
      if (cert.bits & pair_bit(cert.n, i, j)) out.emplace_back(i, j);
    }
  }
  return out;
}

std::vector<std::vector<int>> automorphisms(int n, const std::vector<Edge>& edges) {
  const Adjacency adj = adjacency(n, edges);
  Colors colors{};
  refine(n, adj, colors);

  // Assign vertices in an order where each one (after the first of its
  // component) is adjacent to one already assigned, so adjacency prunes early.
  std::vector<int> order;
  std::uint32_t placed = 0;
  while (static_cast<int>(order.size()) < n) {
    int next = -1;
    for (int v = 0; v < n && next < 0; ++v) {
      if (!(placed >> v & 1u) && (order.empty() || (adj[v] & placed))) next = v;
    }
    if (next < 0) {
      for (int v = 0; v < n; ++v) {
        if (!(placed >> v & 1u)) {
          next = v;
          break;
        }
      }
    }
    order.push_back(next);
    placed |= 1u << next;
  }

  std::vector<std::vector<int>> out;
  std::vector<int> perm(n, -1);
  std::uint32_t used = 0;
  std::function<void(int)> rec = [&](int k) {
    if (k == n) {
      out.push_back(perm);
      return;
    }
    const int v = order[k];
    for (int u = 0; u < n; ++u) {
      if ((used >> u & 1u) || colors[u] != colors[v]) continue;
      bool ok = true;
      for (int j = 0; j < k && ok; ++j) {
        const int w = order[j];
        ok = ((adj[v] >> w & 1u) != 0) == ((adj[u] >> perm[w] & 1u) != 0);
      }
      if (!ok) continue;
      perm[v] = u;
      used |= 1u << u;
      rec(k + 1);
      used &= ~(1u << u);
      perm[v] = -1;
    }
  };
  rec(0);
  return out;
}

int double_cover_matching(int n, const std::vector<Edge>& edges) {
  const Adjacency adj = adjacency(n, edges);
  std::array<int, kMaxVertices> owner;  // owner[v] = u when u_out is matched to v_in
  owner.fill(-1);
  std::uint32_t visited = 0;
  std::function<bool(int)> augment = [&](int u) {
    for (std::uint32_t m = adj[u] & ~visited; m; m &= m - 1) {
      const int v = __builtin_ctz(m);
      if (visited >> v & 1u) continue;
      visited |= 1u << v;
      if (owner[v] < 0 || augment(owner[v])) {
        owner[v] = u;
        return true;
      }
    }
    return false;
  };
  int size = 0;
  for (int u = 0; u < n; ++u) {
    visited = 0;
    size += augment(u);
  }
  return size;
}

}  // namespace nlce
