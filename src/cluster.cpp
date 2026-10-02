#include "cluster.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace nlce {

namespace {

constexpr int L = kLanes;
using Vec = std::vector<u64>;

// A perturbative vector series: v[k] holds [t * L + l] over the first len[k]
// states (times `width` columns for the effective-Hamiltonian chains).
struct Chain {
  std::vector<Vec> v;
  std::vector<std::uint32_t> len;
};

std::vector<int> graph_distances(int nv, const std::vector<Edge>& edges) {
  std::vector<int> d(nv * nv, -1);
  std::vector<std::vector<int>> adj(nv);
  for (auto [i, j] : edges) {
    adj[i].push_back(j);
    adj[j].push_back(i);
  }
  for (int s = 0; s < nv; ++s) {
    std::vector<int> frontier{s};
    d[s * nv + s] = 0;
    for (int k = 1; !frontier.empty(); ++k) {
      std::vector<int> next;
      for (int u : frontier) {
        for (int w : adj[u]) {
          if (d[s * nv + w] < 0) {
            d[s * nv + w] = k;
            next.push_back(w);
          }
        }
      }
      frontier.swap(next);
    }
  }
  for (int x : d) {
    if (x < 0) throw std::invalid_argument("cluster graph is not connected");
  }
  return d;
}

class Engine {
public:
  Engine(const ClusterPlan& plan, const LaneBlock& lanes, RawSeries& out)
      : P_(plan), lb_(lanes), out_(out), nv_(plan.in.nv), ng_(plan.in.ng), nc_(plan.in.nc) {}

  void run() {
    const std::size_t npair = static_cast<std::size_t>(nv_) * nv_;
    const std::size_t npat = P_.in.patterns.size();
    out_.nv = nv_;
    out_.ng = ng_;
    out_.nc = nc_;
    out_.npat = static_cast<int>(npat);
    out_.E.assign((ng_ + 1) * L, 0);
    out_.Hp.assign(npair * (ng_ + 1) * L, 0);
    out_.Hh.assign(npair * (ng_ + 1) * L, 0);
    out_.corr.assign(npair * (ng_ + 1) * L, 0);
    out_.chi.assign(npat * (nc_ + 1) * L, 0);
    out_.m0.assign(npat * (nc_ + 1) * L, 0);

    inv_mott_ = inverse_denominators(P_.mott, 0);
    ground_chain(false, R_, out_.E);
    Vec left_energy((ng_ + 1) * L, 0);
    ground_chain(true, L_, left_energy);
    if (left_energy != out_.E) throw std::logic_error("left and right ground-state energies differ");

    normalisation();
    correlations();
    for (std::size_t q = 0; q < npat; ++q) decoration(static_cast<int>(q));
    R_ = {};
    L_ = {};
    effective_hamiltonian(P_.particle, 1, out_.Hp);
    effective_hamiltonian(P_.hole, 0, out_.Hh);
  }

private:
  const ClusterPlan& P_;
  const LaneBlock& lb_;
  RawSeries& out_;
  const int nv_, ng_, nc_;
  Vec inv_mott_;
  Chain R_, L_;     // right and left ground-state vectors
  Vec norm_inv_;    // series 1/<phi|psi>, [k][l]

  const Modulus& M(int l) const { return lb_.mod[l]; }

  // 1/(E_D - E0(t)) per energy class and lane.  Classes used only by seed
  // states are never inverted; any other vanishing denominator is a genuine
  // degeneracy of H0, which the perturbation theory cannot handle.
  Vec inverse_denominators(const Sector& sec, int ed) const {
    std::vector<char> used(sec.energies.size(), 0);
    for (std::uint32_t t = sec.nseeds; t < sec.size(); ++t) used[sec.energy_class[t]] = 1;
    Vec inv(sec.energies.size() * L, 0);
    for (std::size_t c = 0; c < sec.energies.size(); ++c) {
      if (!used[c]) continue;
      auto [onsite, bonds] = sec.energies[c];
      for (int l = 0; l < L; ++l) {
        const Rational& v = lb_.v[l];
        const i64 num = static_cast<i64>(ed - onsite) * v.den - v.num * bonds;
        if (num == 0) throw std::domain_error("degenerate energy denominator");
        inv[c * L + l] = M(l).mul(M(l).from_int(v.den), M(l).inv(M(l).from_int(num)));
      }
    }
    return inv;
  }

  void ground_chain(bool transpose, Chain& C, Vec& energy) {
    const Sector& S = P_.mott;
    C.v.assign(ng_ + 1, {});
    C.len.assign(ng_ + 1, 0);
    C.len[0] = 1;
    C.v[0].assign(L, 1);
    std::fill(energy.begin(), energy.end(), 0);
    for (int k = 1; k <= ng_; ++k) {
      const std::uint32_t n = S.len(P_.lim_gs[k]);
      const std::uint32_t np = C.len[k - 1];
      const Vec& prev = C.v[k - 1];
      Vec& cur = C.v[k];
      cur.assign(static_cast<std::size_t>(n) * L, 0);
      C.len[k] = n;
      for (std::uint32_t t = 0; t < n; ++t) {
        u64 acc[L] = {};
        for (std::uint32_t m = S.move_begin[t]; m < S.move_begin[t + 1]; ++m) {
          const Move& mv = S.moves[m];
          if (mv.s >= np) break;
          const u64 c = transpose ? mv.bwd : mv.fwd;
          const u64* x = &prev[static_cast<std::size_t>(mv.s) * L];
          for (int l = 0; l < L; ++l) acc[l] += c * x[l];
        }
        if (t == 0) {
          for (int l = 0; l < L; ++l) energy[k * L + l] = M(l).reduce(acc[l]);
          continue;
        }
        u128 sub[L] = {};
        for (int m = 1; m < k; ++m) {
          if (t >= C.len[k - m]) continue;
          const u64* y = &C.v[k - m][static_cast<std::size_t>(t) * L];
          const u64* e = &energy[m * L];
          for (int l = 0; l < L; ++l) sub[l] += static_cast<u128>(e[l]) * y[l];
        }
        const u64* inv = &inv_mott_[S.energy_class[t] * L];
        u64* dst = &cur[static_cast<std::size_t>(t) * L];
        for (int l = 0; l < L; ++l) {
          dst[l] = M(l).mul(M(l).sub(M(l).reduce(acc[l]), M(l).reduce(sub[l])), inv[l]);
        }
      }
    }
  }

  // sum_t a[t] b[t] over the first n states, per lane.
  void dot(const Vec& a, const Vec& b, std::uint32_t n, u64* out) const {
    u128 acc[L] = {};
    for (std::uint32_t t = 0; t < n; ++t) {
      const u64* x = &a[static_cast<std::size_t>(t) * L];
      const u64* y = &b[static_cast<std::size_t>(t) * L];
      for (int l = 0; l < L; ++l) acc[l] += static_cast<u128>(x[l]) * y[l];
    }
    for (int l = 0; l < L; ++l) out[l] = M(l).reduce(acc[l]);
  }

  // out[k] = sum_m num[m] * norm_inv[k-m], k <= order.
  void divide_by_norm(const u64* num, int order, u64* out) const {
    for (int k = 0; k <= order; ++k) {
      u128 acc[L] = {};
      for (int m = 0; m <= k; ++m) {
        for (int l = 0; l < L; ++l) acc[l] += static_cast<u128>(num[m * L + l]) * norm_inv_[(k - m) * L + l];
      }
      for (int l = 0; l < L; ++l) out[k * L + l] = M(l).reduce(acc[l]);
    }
  }

  void normalisation() {
    Vec norm((ng_ + 1) * L, 0);
    u64 tmp[L];
    for (int n = 0; n <= ng_; ++n) {
      for (int a = 0; a <= n; ++a) {
        dot(L_.v[a], R_.v[n - a], std::min(L_.len[a], R_.len[n - a]), tmp);
        for (int l = 0; l < L; ++l) norm[n * L + l] = M(l).add(norm[n * L + l], tmp[l]);
      }
    }
    norm_inv_.assign((ng_ + 1) * L, 0);
    for (int l = 0; l < L; ++l) {
      const u64 i0 = M(l).inv(norm[l]);
      norm_inv_[l] = i0;
      for (int n = 1; n <= ng_; ++n) {
        u128 s = 0;
        for (int k = 1; k <= n; ++k) s += static_cast<u128>(norm[k * L + l]) * norm_inv_[(n - k) * L + l];
        norm_inv_[n * L + l] = M(l).neg(M(l).mul(M(l).reduce(s), i0));
      }
    }
  }

  // num[(u,w)][n] = sum_{a+b=n} <phi_a| b^dag_u b_w + b^dag_w b_u |psi_b>.
  void correlations() {
    if (nv_ < 2) return;
    const Sector& S = P_.mott;
    const std::size_t npair = static_cast<std::size_t>(nv_) * nv_;
    std::vector<u128> num(npair * (ng_ + 1) * L, 0);
    auto fold = [&] {
      for (std::size_t i = 0; i < num.size(); ++i) num[i] = M(static_cast<int>(i % L)).reduce(num[i]);
    };
    const std::uint32_t nstates = static_cast<std::uint32_t>(P_.pair_begin.size()) - 1;
    for (std::uint32_t s = 0; s < nstates; ++s) {
      const int ds = S.dist[s];
      for (std::uint32_t q = P_.pair_begin[s]; q < P_.pair_begin[s + 1]; ++q) {
        const PairMove& pm = P_.pairs[q];
        const int d2 = S.dist[pm.s2];
        const std::size_t base = static_cast<std::size_t>(pm.pair) * (ng_ + 1);
        for (int b = ds; b <= ng_ && d2 <= ng_ - b; ++b) {
          if (s >= R_.len[b]) continue;
          const u64* x = &R_.v[b][static_cast<std::size_t>(s) * L];
          u64 cx[L];
          for (int l = 0; l < L; ++l) cx[l] = pm.coef * x[l];
          for (int a = d2; a <= ng_ - b; ++a) {
            if (pm.s2 >= L_.len[a]) continue;
            const u64* y = &L_.v[a][static_cast<std::size_t>(pm.s2) * L];
            u128* acc = &num[(base + a + b) * L];
            for (int l = 0; l < L; ++l) acc[l] += static_cast<u128>(cx[l]) * y[l];
          }
        }
      }
      if ((s & 1023u) == 1023u) fold();
    }
    fold();
    Vec series((ng_ + 1) * L);
    for (int u = 0; u < nv_; ++u) {
      for (int w = u + 1; w < nv_; ++w) {
        const std::size_t base = (static_cast<std::size_t>(u) * nv_ + w) * (ng_ + 1) * L;
        for (std::size_t i = 0; i < series.size(); ++i) series[i] = static_cast<u64>(num[base + i]);
        divide_by_norm(series.data(), ng_, &out_.corr[out_.pair_at(u, w, 0)]);
        std::copy_n(&out_.corr[out_.pair_at(u, w, 0)], (ng_ + 1) * L, &out_.corr[out_.pair_at(w, u, 0)]);
      }
    }
  }

  // K x (or K^T x) over the first n states, for current pattern `pat`.
  void apply_current(const Pattern& pat, bool transpose, const Vec& x, std::uint32_t nx, std::uint32_t n,
                     Vec& y) const {
    const Sector& S = P_.mott;
    y.assign(static_cast<std::size_t>(n) * L, 0);
    for (std::uint32_t t = 0; t < n; ++t) {
      u64 pos[L] = {}, neg[L] = {};
      for (std::uint32_t m = S.move_begin[t]; m < S.move_begin[t + 1]; ++m) {
        const Move& mv = S.moves[m];
        if (mv.s >= nx) break;
        int sign = mv.orient * pat[mv.bond];
        if (transpose) sign = -sign;
        const u64 c = transpose ? mv.bwd : mv.fwd;
        const u64* src = &x[static_cast<std::size_t>(mv.s) * L];
        u64* acc = sign > 0 ? pos : neg;
        for (int l = 0; l < L; ++l) acc[l] += c * src[l];
      }
      u64* dst = &y[static_cast<std::size_t>(t) * L];
      for (int l = 0; l < L; ++l) dst[l] = M(l).sub(M(l).reduce(pos[l]), M(l).reduce(neg[l]));
    }
  }

  void decoration(int q) {
    const Sector& S = P_.mott;
    const Pattern& pat = P_.in.patterns[q];
    auto reach = [&](int k) { return std::min(k, nc_ - k) + 1; };

    std::vector<Vec> KR(nc_ + 1), KL(nc_ + 1);
    std::vector<std::uint32_t> klen(nc_ + 1);
    for (int k = 0; k <= nc_; ++k) {
      klen[k] = S.len(reach(k));
      apply_current(pat, false, R_.v[k], R_.len[k], klen[k], KR[k]);
      apply_current(pat, true, L_.v[k], L_.len[k], klen[k], KL[k]);
    }

    // First-order response to g K: psi_g at each order of x.
    std::vector<Vec> Q(nc_ + 1);
    for (int n = 0; n <= nc_; ++n) {
      const std::uint32_t len = klen[n];
      Q[n].assign(static_cast<std::size_t>(len) * L, 0);
      for (std::uint32_t t = 0; t < len; ++t) {
        u64 acc[L] = {};
        if (n > 0) {
          for (std::uint32_t m = S.move_begin[t]; m < S.move_begin[t + 1]; ++m) {
            const Move& mv = S.moves[m];
            if (mv.s >= klen[n - 1]) break;
            const u64* x = &Q[n - 1][static_cast<std::size_t>(mv.s) * L];
            for (int l = 0; l < L; ++l) acc[l] += mv.fwd * x[l];
          }
        }
        u128 sub[L] = {};
        for (int m = 1; m <= n; ++m) {
          if (t >= klen[n - m]) continue;
          const u64* y = &Q[n - m][static_cast<std::size_t>(t) * L];
          for (int l = 0; l < L; ++l) sub[l] += static_cast<u128>(out_.E[m * L + l]) * y[l];
        }
        const u64* kr = &KR[n][static_cast<std::size_t>(t) * L];
        u64 val[L];
        for (int l = 0; l < L; ++l) {
          val[l] = M(l).add(M(l).sub(M(l).reduce(acc[l]), M(l).reduce(sub[l])), kr[l]);
        }
        if (t == 0) {
          for (int l = 0; l < L; ++l) {
            if (val[l]) throw std::logic_error("odd-order current response of the energy does not vanish");
          }
          continue;
        }
        const u64* inv = &inv_mott_[S.energy_class[t] * L];
        u64* dst = &Q[n][static_cast<std::size_t>(t) * L];
        for (int l = 0; l < L; ++l) dst[l] = M(l).mul(val[l], inv[l]);
      }
    }

    Vec c2((nc_ + 1) * L, 0), m0((nc_ + 1) * L, 0);
    u64 tmp[L];
    for (int n = 0; n <= nc_; ++n) {
      for (int a = 0; a <= n; ++a) {
        const std::uint32_t overlap = std::min(klen[a], klen[n - a]);
        dot(KL[a], Q[n - a], overlap, tmp);
        for (int l = 0; l < L; ++l) c2[n * L + l] = M(l).add(c2[n * L + l], tmp[l]);
        dot(KL[a], KR[n - a], overlap, tmp);
        for (int l = 0; l < L; ++l) m0[n * L + l] = M(l).sub(m0[n * L + l], tmp[l]);
      }
    }
    divide_by_norm(c2.data(), nc_, &out_.chi[out_.pattern_at(q, 0)]);
    divide_by_norm(m0.data(), nc_, &out_.m0[out_.pattern_at(q, 0)]);
  }

  // Bloch effective Hamiltonian in the one-quasiparticle manifold
  // {seed j}, with E_D = ed, minus the cluster Mott energy.
  void effective_hamiltonian(const Sector& S, int ed, Vec& H) {
    const int nv = nv_;
    const Vec inv = inverse_denominators(S, ed);
    auto at = [&](int i, int j, int k) { return ((static_cast<std::size_t>(i) * nv + j) * (ng_ + 1) + k) * L; };

    std::vector<Vec> psi(ng_ + 1);
    std::vector<std::uint32_t> len(ng_ + 1, 0);
    len[0] = S.len(0);
    psi[0].assign(static_cast<std::size_t>(len[0]) * nv * L, 0);
    for (int j = 0; j < nv; ++j) {
      for (int l = 0; l < L; ++l) {
        psi[0][(static_cast<std::size_t>(j) * nv + j) * L + l] = 1;
        H[at(j, j, 0) + l] = static_cast<u64>(ed);
      }
    }

    u64 acc[kMaxVertices][L];
    for (int k = 1; k <= ng_; ++k) {
      const std::uint32_t n = S.len(std::min(k, ng_ - k));
      const bool keep = k < ng_;
      len[k] = n;
      if (keep) psi[k].assign(static_cast<std::size_t>(n) * nv * L, 0);
      const Vec& prev = psi[k - 1];
      for (std::uint32_t t = 0; t < n; ++t) {
        const std::uint8_t* sd = &S.seed_dist[static_cast<std::size_t>(t) * nv];
        std::uint32_t active = 0;
        for (int j = 0; j < nv; ++j) {
          if (sd[j] <= k) active |= 1u << j;
        }
        if (!active) continue;
        std::uint32_t mend = S.move_begin[t];
        while (mend < S.move_begin[t + 1] && S.moves[mend].s < len[k - 1]) ++mend;
        for (std::uint32_t a = active; a; a &= a - 1) {
          const int j = __builtin_ctz(a);
          u64 r[L] = {};
          for (std::uint32_t m = S.move_begin[t]; m < mend; ++m) {
            const Move& mv = S.moves[m];
            const u64 c = mv.fwd;
            const u64* xj = &prev[(static_cast<std::size_t>(mv.s) * nv + j) * L];
            for (int l = 0; l < L; ++l) r[l] += c * xj[l];
          }
          std::copy_n(r, L, acc[j]);
        }
        if (t < static_cast<std::uint32_t>(S.nseeds)) {
          for (std::uint32_t a = active; a; a &= a - 1) {
            const int j = __builtin_ctz(a);
            for (int l = 0; l < L; ++l) H[at(t, j, k) + l] = M(l).reduce(acc[j][l]);
          }
          continue;
        }
        if (!keep) continue;
        const u64* iv = &inv[S.energy_class[t] * L];
        for (std::uint32_t a = active; a; a &= a - 1) {
          const int j = __builtin_ctz(a);
          u128 sub[L] = {};
          for (int m = 1; m < k; ++m) {
            if (t >= len[k - m]) continue;
            const u64* row = &psi[k - m][static_cast<std::size_t>(t) * nv * L];
            for (int i = 0; i < nv; ++i) {
              if (sd[i] > k - m || P_.graph_dist[i * nv + j] > m) continue;
              const u64* h = &H[at(i, j, m)];
              const u64* y = row + i * L;
              for (int l = 0; l < L; ++l) sub[l] += static_cast<u128>(h[l]) * y[l];
            }
          }
          u64* dst = &psi[k][(static_cast<std::size_t>(t) * nv + j) * L];
          for (int l = 0; l < L; ++l) {
            dst[l] = M(l).mul(M(l).sub(M(l).reduce(acc[j][l]), M(l).reduce(sub[l])), iv[l]);
          }
        }
      }
    }
    for (int j = 0; j < nv; ++j) {
      for (int k = 0; k <= ng_; ++k) {
        for (int l = 0; l < L; ++l) H[at(j, j, k) + l] = M(l).sub(H[at(j, j, k) + l], out_.E[k * L + l]);
      }
    }
  }
};

}  // namespace

ClusterPlan plan_cluster(ClusterInput in) {
  ClusterPlan plan;
  const int nv = in.nv, ng = in.ng, nc = in.nc;
  if (nv < 1 || nv > kMaxVertices) throw std::invalid_argument("cluster size out of range");
  if (ng < 1 || nc < 0 || nc > ng) throw std::invalid_argument("bad series orders");
  for (const auto& p : in.patterns) {
    if (p.size() != in.edges.size()) throw std::invalid_argument("pattern does not match edges");
  }
  plan.graph_dist = graph_distances(nv, in.edges);
  plan.diameter = *std::max_element(plan.graph_dist.begin(), plan.graph_dist.end());

  // Ground-state vectors feed: the energy (reach 0, order ng), the pair
  // correlators (reach = diameter, order ng), and the current operators
  // (reach 2, order nc).
  const int bound = std::max(ng + plan.diameter, nc + 2);
  plan.lim_gs.resize(ng + 1);
  int dmax = 0;
  for (int k = 0; k <= ng; ++k) {
    plan.lim_gs[k] = std::min(k, bound - k);
    dmax = std::max(dmax, plan.lim_gs[k]);
  }
  dmax = std::max(dmax, std::min(nc / 2 + 1, ng));

  const u64 mott = uniform_state(nv, 1);
  std::vector<u64> doublons, holons;
  for (int j = 0; j < nv; ++j) {
    doublons.push_back(mott + site_unit(j));
    holons.push_back(mott - site_unit(j));
  }
  plan.mott = build_sector(nv, in.edges, {mott}, dmax, false);
  plan.particle = build_sector(nv, in.edges, doublons, ng / 2, true);
  plan.hole = build_sector(nv, in.edges, holons, ng / 2, true);

  const Sector& S = plan.mott;
  const std::uint32_t nstates = S.len((ng + plan.diameter) / 2);
  plan.pair_begin.reserve(nstates + 1);
  plan.pair_begin.push_back(0);
  for (std::uint32_t s = 0; s < nstates; ++s) {
    const u64 st = S.states[s];
    for (int from = 0; from < nv; ++from) {
      const int n = occ(st, from);
      if (!n) continue;
      for (int to = 0; to < nv; ++to) {
        if (to == from || occ(st, to) == 15) continue;
        auto it = S.index.find(st - site_unit(from) + site_unit(to));
        if (it == S.index.end()) continue;
        const int u = std::min(from, to), w = std::max(from, to);
        plan.pairs.push_back({it->second, static_cast<std::uint16_t>(u * nv + w), static_cast<std::uint8_t>(n)});
      }
    }
    plan.pair_begin.push_back(static_cast<std::uint32_t>(plan.pairs.size()));
  }
  plan.in = std::move(in);
  return plan;
}

void compute_block(const ClusterPlan& plan, const LaneBlock& lanes, RawSeries& out) {
  Engine(plan, lanes, out).run();
}

}  // namespace nlce
