#include "pipeline.hpp"

#include "cluster.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace nlce {

namespace {

constexpr int L = kLanes;

// Cumulant orders k0..order of one class (or key), all lanes of the pass:
// data[(q * nk + k - k0) * nlanes + lane].
struct Weights {
  int k0 = 0;
  int nk = 0;
  std::vector<u64> data;
  std::size_t at(int q, int k, std::size_t nlanes) const {
    return (static_cast<std::size_t>(q) * nk + (k - k0)) * nlanes;
  }
};

class Pass {
public:
  Pass(const Geometry& geo, int ng, int nc, const std::vector<LaneSpec>& lanes, const PassOptions& opts)
      : geo_(geo), ng_(ng), nc_(nc), lanes_(lanes), opts_(opts), nl_(lanes.size()) {
    layout_ = {ng, nc, static_cast<int>(geo.displacements.size())};
    const std::size_t nb = (nl_ + L - 1) / L;
    blocks_.resize(nb);
    for (std::size_t b = 0; b < nb; ++b) {
      LaneBlock& blk = blocks_[b];
      blk.count = static_cast<int>(std::min<std::size_t>(L, nl_ - b * L));
      for (int l = 0; l < L; ++l) {
        const LaneSpec& spec = lanes_[b * L + std::min(l, blk.count - 1)];
        blk.mod[l] = Modulus(spec.p);
        blk.v[l] = spec.v;
        blk.vres[l] = blk.mod[l].mul(blk.mod[l].from_int(spec.v.num), blk.mod[l].inv(blk.mod[l].from_int(spec.v.den)));
      }
    }
    class_w_.resize(geo.classes.size());
    key_w_.resize(geo.keys.size());
  }

  std::vector<std::vector<u64>> run() {
    const int threads = std::max(1, opts_.threads);
    std::vector<std::vector<u64>> totals(threads, std::vector<u64>(nl_ * layout_.size(), 0));
    const auto start = std::chrono::steady_clock::now();
    for (int s = 1; s <= geo_.smax; ++s) {
      std::vector<int> wave;
      for (int c = 0; c < static_cast<int>(geo_.classes.size()); ++c) {
        if (geo_.classes[c].nv == s) wave.push_back(c);
      }
      std::stable_sort(wave.begin(), wave.end(), [&](int x, int y) {
        return geo_.classes[x].edges.size() > geo_.classes[y].edges.size();
      });
      std::atomic<std::size_t> next{0};
      std::exception_ptr error;
      std::mutex error_mutex;
      std::atomic<bool> failed{false};
      {
        std::vector<std::jthread> pool;
        for (int t = 0; t < threads; ++t) {
          pool.emplace_back([&, t] {
            while (!failed) {
              const std::size_t i = next++;
              if (i >= wave.size()) break;
              try {
                process(wave[i], totals[t]);
              } catch (...) {
                std::lock_guard lock(error_mutex);
                if (!error) error = std::current_exception();
                failed = true;
              }
            }
          });
        }
      }
      if (error) std::rethrow_exception(error);
      if (opts_.verbose) {
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "    s=%2d  %6zu classes  %8.1fs\n", s, wave.size(), sec);
      }
    }

    std::vector<std::vector<u64>> out(nl_, std::vector<u64>(layout_.size(), 0));
    for (std::size_t g = 0; g < nl_; ++g) {
      const Modulus& m = blocks_[g / L].mod[g % L];
      for (std::size_t i = 0; i < layout_.size(); ++i) {
        u64 acc = 0;
        for (const auto& tot : totals) acc = m.add(acc, tot[g * layout_.size() + i]);
        out[g][i] = acc;
      }
    }
    // The cumulants carry H_eff minus the seed energy; restore E_D = 1 for
    // the particle on the zero displacement.
    const auto zero = std::find(geo_.displacements.begin(), geo_.displacements.end(), Site{0, 0});
    const std::size_t cd0 = zero - geo_.displacements.begin();
    for (std::size_t g = 0; g < nl_; ++g) {
      const Modulus& m = blocks_[g / L].mod[g % L];
      u64& x = out[g][layout_.hp() + cd0 * (ng_ + 1)];
      x = m.add(x, 1);
    }
    return out;
  }

private:
  const Geometry& geo_;
  const int ng_, nc_;
  const std::vector<LaneSpec>& lanes_;
  const PassOptions& opts_;
  const std::size_t nl_;
  SeriesLayout layout_;
  std::vector<LaneBlock> blocks_;
  std::vector<Weights> class_w_;
  std::vector<Weights> key_w_;

  void process(int c, std::vector<u64>& totals) {
    const ClassInfo& cls = geo_.classes[c];
    const int nv = cls.nv;
    const int s = nv;
    const std::size_t npair = static_cast<std::size_t>(nv) * nv;

    ClusterInput input{nv, cls.edges, {}, ng_, nc_};
    for (int key : cls.keys) input.patterns.push_back(geo_.keys[key].pattern);
    const ClusterPlan plan = plan_cluster(std::move(input));
    const std::vector<SubCluster> subs = subclusters(geo_, c);
    std::vector<std::vector<int>> subkeys(cls.keys.size());
    for (std::size_t q = 0; q < cls.keys.size(); ++q) {
      for (const auto& sub : subs) subkeys[q].push_back(subcluster_key(geo_, c, sub, geo_.keys[cls.keys[q]].pattern));
    }

    const int k0 = s - 1;
    const int k0c = std::max(0, s - 2);
    const bool keep = s < geo_.smax;
    Weights* cw = nullptr;
    if (keep) {
      cw = &class_w_[c];
      cw->k0 = k0;
      cw->nk = ng_ - k0 + 1;
      cw->data.assign((1 + 3 * npair) * cw->nk * nl_, 0);
      for (int key : cls.keys) {
        Weights& kw = key_w_[key];
        kw.k0 = k0c;
        kw.nk = nc_ - k0c + 1;
        kw.data.assign(2 * kw.nk * nl_, 0);
      }
    }

    RawSeries raw;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
      const LaneBlock& blk = blocks_[b];
      const std::size_t g0 = b * L;
      compute_block(plan, blk, raw);

      for (int u = 0; u < nv; ++u) {
        for (int l = 0; l < blk.count; ++l) {
          u64& x = raw.Hp[raw.pair_at(u, u, 0) + l];
          x = blk.mod[l].sub(x, 1);
        }
      }

      for (const auto& sub : subs) {
        const Weights& w = class_w_[sub.cls];
        const int nv2 = static_cast<int>(sub.verts.size());
        const int np2 = nv2 * nv2;
        for (int k = w.k0; k <= ng_; ++k) {
          subtract(blk, &raw.E[k * L], &w.data[w.at(0, k, nl_) + g0]);
          for (int x = 0; x < nv2; ++x) {
            for (int y = 0; y < nv2; ++y) {
              const int u = sub.verts[x], v = sub.verts[y];
              const int q = sub.map[x] * nv2 + sub.map[y];
              subtract(blk, &raw.Hp[raw.pair_at(u, v, k)], &w.data[w.at(1 + q, k, nl_) + g0]);
              subtract(blk, &raw.Hh[raw.pair_at(u, v, k)], &w.data[w.at(1 + np2 + q, k, nl_) + g0]);
              if (x != y) subtract(blk, &raw.corr[raw.pair_at(u, v, k)], &w.data[w.at(1 + 2 * np2 + q, k, nl_) + g0]);
            }
          }
        }
      }
      for (std::size_t q = 0; q < cls.keys.size(); ++q) {
        for (std::size_t i = 0; i < subs.size(); ++i) {
          const Weights& w = key_w_[subkeys[q][i]];
          for (int k = w.k0; k <= nc_; ++k) {
            subtract(blk, &raw.chi[raw.pattern_at(static_cast<int>(q), k)], &w.data[w.at(0, k, nl_) + g0]);
            subtract(blk, &raw.m0[raw.pattern_at(static_cast<int>(q), k)], &w.data[w.at(1, k, nl_) + g0]);
          }
        }
      }

      check_vanishing(blk, raw, c, k0, k0c);

      if (keep) {
        for (int k = k0; k <= ng_; ++k) {
          store(blk, &raw.E[k * L], &cw->data[cw->at(0, k, nl_) + g0]);
          for (std::size_t q = 0; q < npair; ++q) {
            const int u = static_cast<int>(q) / nv, v = static_cast<int>(q) % nv;
            store(blk, &raw.Hp[raw.pair_at(u, v, k)], &cw->data[cw->at(1 + q, k, nl_) + g0]);
            store(blk, &raw.Hh[raw.pair_at(u, v, k)], &cw->data[cw->at(1 + npair + q, k, nl_) + g0]);
            store(blk, &raw.corr[raw.pair_at(u, v, k)], &cw->data[cw->at(1 + 2 * npair + q, k, nl_) + g0]);
          }
        }
        for (std::size_t q = 0; q < cls.keys.size(); ++q) {
          Weights& kw = key_w_[cls.keys[q]];
          for (int k = k0c; k <= nc_; ++k) {
            store(blk, &raw.chi[raw.pattern_at(static_cast<int>(q), k)], &kw.data[kw.at(0, k, nl_) + g0]);
            store(blk, &raw.m0[raw.pattern_at(static_cast<int>(q), k)], &kw.data[kw.at(1, k, nl_) + g0]);
          }
        }
      }

      accumulate(blk, g0, raw, cls, k0, k0c, totals);
    }
  }

  static void subtract(const LaneBlock& blk, u64* dst, const u64* w) {
    for (int l = 0; l < blk.count; ++l) dst[l] = blk.mod[l].sub(dst[l], w[l]);
  }

  static void store(const LaneBlock& blk, const u64* src, u64* dst) { std::copy_n(src, blk.count, dst); }

  void check_vanishing(const LaneBlock& blk, const RawSeries& raw, int c, int k0, int k0c) const {
    const int nv = raw.nv;
    auto zero = [&](const u64* x) {
      for (int l = 0; l < blk.count; ++l) {
        if (x[l]) return false;
      }
      return true;
    };
    bool ok = true;
    for (int k = 0; k < k0 && ok; ++k) {
      ok = zero(&raw.E[k * L]);
      for (int u = 0; u < nv && ok; ++u) {
        for (int v = 0; v < nv && ok; ++v) {
          ok = zero(&raw.Hp[raw.pair_at(u, v, k)]) && zero(&raw.Hh[raw.pair_at(u, v, k)]) &&
               zero(&raw.corr[raw.pair_at(u, v, k)]);
        }
      }
    }
    for (int q = 0; q < raw.npat && ok; ++q) {
      for (int k = 0; k < k0c && ok; ++k) ok = zero(&raw.chi[raw.pattern_at(q, k)]) && zero(&raw.m0[raw.pattern_at(q, k)]);
    }
    if (!ok) {
      throw std::logic_error("cluster cumulant does not vanish below its leading order (class " + std::to_string(c) +
                             ", " + std::to_string(nv) + " sites)");
    }
  }

  void accumulate(const LaneBlock& blk, std::size_t g0, const RawSeries& raw, const ClassInfo& cls, int k0, int k0c,
                  std::vector<u64>& totals) const {
    const std::size_t width = layout_.size();
    for (int l = 0; l < blk.count; ++l) {
      const Modulus& m = blk.mod[l];
      u64* tot = &totals[(g0 + l) * width];
      const u64 inv12 = m.inv(12);
      const u64 mult = m.from_int(cls.mult);
      for (int k = k0; k <= ng_; ++k) tot[layout_.en() + k] = m.add(tot[layout_.en() + k], m.mul(mult, raw.E[k * L + l]));
      for (const Embedding& e : cls.embeddings) {
        const u64 f = m.mul(m.from_int(e.fac12), inv12);
        const std::size_t off = static_cast<std::size_t>(e.cd) * (ng_ + 1);
        for (int k = k0; k <= ng_; ++k) {
          u64& hp = tot[layout_.hp() + off + k];
          hp = m.add(hp, m.mul(f, raw.Hp[raw.pair_at(e.a, e.b, k) + l]));
          u64& hh = tot[layout_.hh() + off + k];
          hh = m.add(hh, m.mul(f, raw.Hh[raw.pair_at(e.a, e.b, k) + l]));
          if (e.a != e.b) {
            u64& sq = tot[layout_.s() + off + k];
            sq = m.add(sq, m.mul(f, raw.corr[raw.pair_at(e.a, e.b, k) + l]));
          }
        }
      }
      for (std::size_t q = 0; q < cls.keys.size(); ++q) {
        const u64 km = m.from_int(geo_.keys[cls.keys[q]].mult);
        const u64 km2 = m.add(km, km);
        for (int k = k0c; k <= nc_; ++k) {
          u64& ch = tot[layout_.chi() + k];
          ch = m.add(ch, m.mul(km2, raw.chi[raw.pattern_at(static_cast<int>(q), k) + l]));
          u64& mm = tot[layout_.m0() + k];
          mm = m.add(mm, m.mul(km, raw.m0[raw.pattern_at(static_cast<int>(q), k) + l]));
        }
      }
    }
  }
};

}  // namespace

std::vector<std::vector<u64>> run_pass(const Geometry& geo, int ng, int nc, const std::vector<LaneSpec>& lanes,
                                       const PassOptions& opts) {
  if (lanes.empty()) return {};
  return Pass(geo, ng, nc, lanes, opts).run();
}

}  // namespace nlce
