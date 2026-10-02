#include "sector.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace nlce {

u64 uniform_state(int nv, int n) {
  u64 st = 0;
  for (int i = 0; i < nv; ++i) st |= static_cast<u64>(n) << (4 * i);
  return st;
}

Sector build_sector(int nv, const std::vector<Edge>& edges, const std::vector<u64>& seeds, int dmax,
                    bool seed_distances) {
  if (nv > kMaxVertices) throw std::invalid_argument("cluster too large for packed states");
  if (edges.size() > std::numeric_limits<std::uint8_t>::max()) throw std::invalid_argument("too many edges");
  if (dmax > std::numeric_limits<std::uint8_t>::max()) throw std::invalid_argument("hop distance too large");
  Sector sec;
  sec.nv = nv;
  sec.nseeds = static_cast<int>(seeds.size());
  sec.dmax = dmax;

  auto add = [&](u64 st, int d) {
    if (sec.index.try_emplace(st, sec.size()).second) {
      sec.states.push_back(st);
      sec.dist.push_back(static_cast<std::uint8_t>(d));
      return true;
    }
    return false;
  };
  for (u64 st : seeds) {
    if (!add(st, 0)) throw std::invalid_argument("duplicate seed");
  }
  sec.layer_end.push_back(sec.size());
  for (int d = 1; d <= dmax; ++d) {
    const std::uint32_t begin = d == 1 ? 0 : sec.layer_end[d - 2];
    const std::uint32_t end = sec.layer_end[d - 1];
    for (std::uint32_t k = begin; k < end; ++k) {
      const u64 st = sec.states[k];
      for (auto [i, j] : edges) {
        for (auto [from, to] : {std::pair{j, i}, std::pair{i, j}}) {
          if (!occ(st, from)) continue;
          if (occ(st, to) == kMaxOccupation) throw std::overflow_error("site occupation exceeds packed range");
          add(st - site_unit(from) + site_unit(to), d);
        }
      }
    }
    sec.layer_end.push_back(sec.size());
  }

  std::map<std::pair<int, int>, int> energy_index;
  sec.energy_class.resize(sec.size());
  for (std::uint32_t t = 0; t < sec.size(); ++t) {
    const u64 st = sec.states[t];
    int onsite = 0, bonds = 0;
    for (int i = 0; i < nv; ++i) onsite += occ(st, i) * (occ(st, i) - 1) / 2;
    for (auto [i, j] : edges) bonds += (occ(st, i) - 1) * (occ(st, j) - 1);
    auto [it, fresh] = energy_index.try_emplace({onsite, bonds}, static_cast<int>(sec.energies.size()));
    if (fresh) {
      if (sec.energies.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::overflow_error("too many distinct unperturbed energies");
      }
      sec.energies.push_back({onsite, bonds});
    }
    sec.energy_class[t] = static_cast<std::uint16_t>(it->second);
  }

  sec.move_begin.reserve(sec.size() + 1);
  sec.move_begin.push_back(0);
  for (std::uint32_t t = 0; t < sec.size(); ++t) {
    const u64 st = sec.states[t];
    const std::size_t first = sec.moves.size();
    for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
      auto [i, j] = edges[e];
      // The boson now on `to` arrived from `from`.
      for (auto [to, from] : {std::pair{i, j}, std::pair{j, i}}) {
        if (!occ(st, to)) continue;
        auto it = sec.index.find(st - site_unit(to) + site_unit(from));
        if (it == sec.index.end()) continue;
        sec.moves.push_back({it->second, static_cast<std::uint8_t>(occ(st, from) + 1),
                             static_cast<std::uint8_t>(occ(st, to)), static_cast<std::uint8_t>(e),
                             static_cast<std::int8_t>(to < from ? 1 : -1)});
      }
    }
    std::sort(sec.moves.begin() + first, sec.moves.end(), [](const Move& x, const Move& y) { return x.s < y.s; });
    sec.move_begin.push_back(static_cast<std::uint32_t>(sec.moves.size()));
  }

  if (seed_distances) {
    const int ns = sec.nseeds;
    sec.seed_dist.assign(static_cast<std::size_t>(sec.size()) * ns, 255);
    std::vector<std::uint32_t> frontier, next;
    for (int q = 0; q < ns; ++q) {
      sec.seed_dist[static_cast<std::size_t>(q) * ns + q] = 0;
      frontier.assign(1, static_cast<std::uint32_t>(q));
      for (int d = 1; !frontier.empty() && d < 255; ++d) {
        next.clear();
        for (std::uint32_t t : frontier) {
          for (std::uint32_t m = sec.move_begin[t]; m < sec.move_begin[t + 1]; ++m) {
            auto& sd = sec.seed_dist[static_cast<std::size_t>(sec.moves[m].s) * ns + q];
            if (sd == 255) {
              sd = static_cast<std::uint8_t>(d);
              next.push_back(sec.moves[m].s);
            }
          }
        }
        frontier.swap(next);
      }
    }
  }
  return sec;
}

}  // namespace nlce
