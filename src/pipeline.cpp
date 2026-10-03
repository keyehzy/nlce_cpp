#include "pipeline.hpp"

#include "cluster.hpp"

#include <boost/unordered/unordered_flat_map.hpp>

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

// Cumulant orders k0..order of one class or key, for every lane of the pass:
// data[(row * nk + k - k0) * nlanes + lane].  Empty when k0 > order, and
// before reset.
class Weights {
public:
  void reset(int rows, int k0, int order, std::size_t nlanes) {
    k0_ = k0;
    nk_ = std::max(0, order - k0 + 1);
    nlanes_ = nlanes;
    data_.assign(static_cast<std::size_t>(rows) * nk_ * nlanes, 0);
  }
  int k0() const { return k0_; }
  int end() const { return k0_ + nk_; }
  u64* at(int row, int k, std::size_t lane) { return data_.data() + offset(row, k, lane); }
  const u64* at(int row, int k, std::size_t lane) const { return data_.data() + offset(row, k, lane); }

private:
  int k0_ = 0;
  int nk_ = 0;
  std::size_t nlanes_ = 0;
  std::vector<u64> data_;

  std::size_t offset(int row, int k, std::size_t lane) const {
    return (static_cast<std::size_t>(row) * nk_ + (k - k0_)) * nlanes_ + lane;
  }
};

// Rows of a class's weights: the energy, then Hp, Hh and corr for each
// vertex pair q = u*nv + w.
struct ClassRows {
  int npair = 0;
  static constexpr int E = 0;
  int Hp(int q) const { return 1 + q; }
  int Hh(int q) const { return 1 + npair + q; }
  int corr(int q) const { return 1 + 2 * npair + q; }
  int count() const { return 1 + 3 * npair; }
};

// Rows of a key's weights.
enum KeyRow { kChiRow = 0, kM0Row = 1, kKeyRows = 2 };

// LEADING ORDERS.  A perturbative term depends only on the sites its hops
// touch, since an untouched site keeps n = 1 and enters neither U nor V, so at
// every order the cumulant of an s-site class collects exactly the terms that
// touch all s sites.  Add to a term's hops its operator insertions (the
// b^dag_u b_w of corr, the two currents of chi and m0) and, for Hp and Hh, one
// arc closing the displacement from seed j to seed i.  The resulting
// multigraph has equal in- and out-degree at every site, and every site
// receives an arc.  Choose one incoming arc per site: these s arcs have at
// most m distinct tails, m = ClassInfo::matching, since one arc per tail is a
// matching of the bipartite double cover; and every site sends at least as
// many arcs as it receives, so there are at least 2s - m arcs.  The closing
// arc or the b^dag b insertion need not be a cluster edge, and raises m by at
// most one, to at most s.  The cumulants therefore vanish below these orders,
// which check_vanishing verifies exactly in every lane.
//
// Idh from (doublon i, holon j) to (i', j') moves one boson net from i to i'
// and one from j' to j.  Every site other than i, j, i', j' receives an arc,
// as do i' when i' != i and j when j != j', which makes at least s - 2 arcs in
// all three cases (|{i, j, i', j'}| = 2, 3, 4).  An adjacent pair interacts
// through V without any hop, so a two-site class starts at order 0.
struct LeadingOrders {
  int energy;   // E
  int pair;     // Hp, Hh, corr
  int current;  // chi, m0
  int dh;       // Idh
};

LeadingOrders leading_orders(const ClassInfo& cls) {
  const int s = cls.nv, m = cls.matching;
  return {2 * s - m, std::max(s - 1, 2 * s - 2 - m), 2 * s - 2 - m, std::max(0, s - 2)};
}

// Rows of a class's Idh weights: dh_row(i', j', i, j) over all vertices.
int dh_rows(int nv) { return nv * nv * nv * nv; }

// OCCUPATION CAP.  In the same picture, a term of order k <= ng has at most
// k + 1 arcs (k + 2 <= nc + 2 for chi and m0), and each of the s sites
// receives at least one, so none receives more than A - s + 1, where
// A = max(ng + 1, nc + 2).  The doublon's seed receives the closing arc, so no
// occupation ever exceeds A - s + 2.  For Idh of order k, the s - 2 required
// arcs leave at most k - s + 2 for any one site, which starts with at most
// two bosons, so A includes ndh + 2.  The terms that touch every site are
// therefore the same in the model capped at A - s + 2 bosons per site, which
// is again local, and its cumulants agree with the full model's through ng
// and nc.  For the largest classes, with ng = smax - 1 and nc = smax - 2, the
// cap is 2, which removes about three quarters of their Mott and particle
// states.  Subtraction works order by order, so the subcluster weights must
// come from the capped model too.  The pass therefore runs two hierarchies:
// the full model for the smaller classes, and the capped model through every
// size, of which only the largest classes enter the lattice sums.
struct Hierarchy {
  int cap = kMaxOccupation;  // bosons per site
  int lo = 1;                // smallest class size entering the lattice sums
  int top = 0;               // largest class size computed
  std::vector<Weights> class_w;
  std::vector<Weights> key_w;
  std::vector<Weights> dh_w;
};

class Pass {
public:
  Pass(const Geometry& geo, int ng, int nc, const std::vector<LaneSpec>& lanes, const PassOptions& opts)
      : geo_(geo), ng_(ng), nc_(nc), lanes_(lanes), opts_(opts), nl_(lanes.size()) {
    layout_ = {ng,
               nc,
               static_cast<int>(geo.displacements.size()),
               opts.ndh,
               static_cast<int>(geo.dh_keys.size()),
               static_cast<int>(geo.pairs.size())};
    if (opts.ndh >= 0 && geo.dh_keys.empty()) throw std::logic_error("doublon-holon keys not built");
    for (std::size_t b = 0; b < nl_; b += L) blocks_.push_back(make_lane_block(lanes_, b));
    const int top = geo.smax;
    const int cap = std::clamp(std::max({ng + 1, nc + 2, opts.ndh + 2}) - top + 2, 2, kMaxOccupation);
    if (opts.cap_largest && cap < kMaxOccupation) {
      hier_.push_back({kMaxOccupation, 1, top - 1, {}, {}, {}});
      hier_.push_back({cap, top, top, {}, {}, {}});
    } else {
      hier_.push_back({kMaxOccupation, 1, top, {}, {}, {}});
    }
    for (Hierarchy& h : hier_) {
      h.class_w.resize(geo.classes.size());
      h.key_w.resize(geo.keys.size());
      h.dh_w.resize(geo.classes.size());
    }
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
      std::vector<std::pair<Hierarchy*, int>> work;
      for (Hierarchy& h : hier_) {
        if (s > h.top) continue;
        for (int c : wave) work.emplace_back(&h, c);
      }
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
              if (i >= work.size()) break;
              try {
                process(*work[i].first, work[i].second, totals[t]);
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
      for (std::size_t pc = 0; pc < geo_.pairs.size(); ++pc) {
        if (!(geo_.pairs[pc][0] == geo_.pairs[pc][1])) continue;
        u64& y = out[g][layout_.hp_pairs() + pc * (ng_ + 1)];
        y = m.add(y, 1);
      }
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
  std::vector<Hierarchy> hier_;

  void process(Hierarchy& h, int c, std::vector<u64>& totals) {
    const ClassInfo& cls = geo_.classes[c];
    const int s = cls.nv;
    const LeadingOrders lead = leading_orders(cls);
    // Cumulants that vanish through ng, nc or ndh are neither computed nor stored.
    const bool decorate = lead.current <= nc_;
    const int ndh = opts_.ndh;
    const bool dh = s >= 2 && lead.dh <= ndh;
    if (lead.pair > ng_ && !decorate && !dh) return;

    ClusterInput input{cls.nv, cls.edges, {}, ng_, nc_, h.cap, dh ? ndh : -1};
    if (decorate) {
      for (int key : cls.keys) input.patterns.push_back(geo_.keys[key].pattern);
    }
    const ClusterPlan plan = plan_cluster(std::move(input));
    const std::vector<SubCluster> subs = subclusters(geo_, c);
    // subkeys[q][i]: key of subcluster i under the class's pattern q.
    std::vector<std::vector<int>> subkeys(decorate ? cls.keys.size() : 0);
    for (std::size_t q = 0; q < subkeys.size(); ++q) {
      for (const auto& sub : subs) subkeys[q].push_back(subcluster_key(geo_, c, sub, geo_.keys[cls.keys[q]].pattern));
    }

    // Clusters of the top size are never subtracted from anything.
    const bool keep = s < h.top;
    if (keep) {
      h.class_w[c].reset(ClassRows{s * s}.count(), lead.pair, ng_, nl_);
      for (int key : cls.keys) h.key_w[key].reset(kKeyRows, lead.current, nc_, nl_);
      if (dh) h.dh_w[c].reset(dh_rows(s), lead.dh, ndh, nl_);
    }
    const std::vector<DhEmbedding> dh_emb = dh && s >= h.lo ? dh_embeddings(cls) : std::vector<DhEmbedding>{};

    RawSeries raw;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
      const LaneBlock& blk = blocks_[b];
      const std::size_t g0 = b * L;
      compute_block(plan, blk, raw);
      remove_seed_energy(blk, raw);
      subtract_subclusters(h, blk, g0, subs, subkeys, raw);
      check_vanishing(blk, raw, c, lead);
      if (keep) store_weights(h, blk, g0, c, raw);
      if (s >= h.lo) accumulate(blk, g0, raw, cls, lead, dh_emb, totals);
    }
  }

  // Lattice weights of the class's Idh rows: per row and key, the
  // realisations' orbits times |G| / |orbit of the key|, as for Embedding.
  // The keys fix the holon's sublattice, so these are not averaged over the
  // unit cell.
  struct DhEmbedding {
    std::uint32_t row;
    int key;
    std::int64_t fac;
  };

  std::vector<DhEmbedding> dh_embeddings(const ClassInfo& cls) const {
    const int n = cls.nv;
    const std::int64_t order = static_cast<std::int64_t>(geo_.lattice->group.size());
    boost::unordered_flat_map<std::uint64_t, std::int64_t> acc;
    std::vector<int> key(n * n * n * n);
    for (const Realization& real : cls.realizations) {
      for (int i2 = 0; i2 < n; ++i2) {
        for (int j2 = 0; j2 < n; ++j2) {
          if (i2 == j2) continue;
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
              if (i == j) continue;
              const int k = geo_.dh_key(real.pos[i2], real.pos[j2], real.pos[i], real.pos[j]);
              const std::uint64_t row = ((static_cast<std::uint64_t>(i2) * n + j2) * n + i) * n + j;
              acc[row << 32 | static_cast<std::uint32_t>(k)] += real.orbit * order / geo_.dh_key_orbit[k];
            }
          }
        }
      }
    }
    std::vector<DhEmbedding> out;
    out.reserve(acc.size());
    for (auto [rk, fac] : acc) {
      out.push_back({static_cast<std::uint32_t>(rk >> 32), static_cast<int>(rk & 0xffffffffu), fac});
    }
    return out;
  }

  // Hp carries the unperturbed doublon energy E_D = 1 on its order-0
  // diagonal, which would spoil the cancellation of the cumulants; run()
  // restores it once in the lattice sum.
  static void remove_seed_energy(const LaneBlock& blk, RawSeries& raw) {
    for (int u = 0; u < raw.nv; ++u) {
      u64* x = &raw.Hp[raw.pair_at(u, u, 0)];
      for (int l = 0; l < blk.count; ++l) x[l] = blk.mod[l].sub(x[l], 1);
    }
  }

  // Subtracts the weights of every connected proper induced subcluster, each
  // mapped from its own canonical labels onto the parent's vertices.
  static void subtract_subclusters(const Hierarchy& h, const LaneBlock& blk, std::size_t g0,
                                   const std::vector<SubCluster>& subs,
                                   const std::vector<std::vector<int>>& subkeys, RawSeries& raw) {
    for (const auto& sub : subs) {
      const Weights& w = h.class_w[sub.cls];
      const int nv2 = static_cast<int>(sub.verts.size());
      const ClassRows rows{nv2 * nv2};
      for (int k = w.k0(); k < w.end(); ++k) {
        subtract(blk, &raw.E[k * L], w.at(ClassRows::E, k, g0));
        for (int x = 0; x < nv2; ++x) {
          for (int y = 0; y < nv2; ++y) {
            const int u = sub.verts[x], v = sub.verts[y];
            const int q = sub.map[x] * nv2 + sub.map[y];
            subtract(blk, &raw.Hp[raw.pair_at(u, v, k)], w.at(rows.Hp(q), k, g0));
            subtract(blk, &raw.Hh[raw.pair_at(u, v, k)], w.at(rows.Hh(q), k, g0));
            if (x != y) subtract(blk, &raw.corr[raw.pair_at(u, v, k)], w.at(rows.corr(q), k, g0));
          }
        }
      }
    }
    if (raw.ndh >= 0) {
      for (const auto& sub : subs) {
        const Weights& w = h.dh_w[sub.cls];
        const int n2 = static_cast<int>(sub.verts.size());
        for (int k = w.k0(); k < w.end(); ++k) {
          for (int a = 0; a < n2; ++a) {
            for (int b = 0; b < n2; ++b) {
              if (a == b) continue;
              for (int c = 0; c < n2; ++c) {
                for (int d = 0; d < n2; ++d) {
                  if (c == d) continue;
                  const int q = ((sub.map[a] * n2 + sub.map[b]) * n2 + sub.map[c]) * n2 + sub.map[d];
                  const std::size_t row = raw.dh_row(sub.verts[a], sub.verts[b], sub.verts[c], sub.verts[d]);
                  subtract(blk, &raw.Idh[raw.dh_at(row, k)], w.at(q, k, g0));
                }
              }
            }
          }
        }
      }
    }
    for (std::size_t q = 0; q < subkeys.size(); ++q) {
      const int pat = static_cast<int>(q);
      for (std::size_t i = 0; i < subs.size(); ++i) {
        const Weights& w = h.key_w[subkeys[q][i]];
        for (int k = w.k0(); k < w.end(); ++k) {
          subtract(blk, &raw.chi[raw.pattern_at(pat, k)], w.at(kChiRow, k, g0));
          subtract(blk, &raw.m0[raw.pattern_at(pat, k)], w.at(kM0Row, k, g0));
        }
      }
    }
  }

  // Keeps the non-vanishing orders of the cumulants for larger clusters.
  void store_weights(Hierarchy& h, const LaneBlock& blk, std::size_t g0, int c, const RawSeries& raw) const {
    const ClassInfo& cls = geo_.classes[c];
    const int nv = cls.nv;
    const ClassRows rows{nv * nv};
    Weights& cw = h.class_w[c];
    for (int k = cw.k0(); k < cw.end(); ++k) {
      store(blk, &raw.E[k * L], cw.at(ClassRows::E, k, g0));
      for (int u = 0; u < nv; ++u) {
        for (int v = 0; v < nv; ++v) {
          const int q = u * nv + v;
          store(blk, &raw.Hp[raw.pair_at(u, v, k)], cw.at(rows.Hp(q), k, g0));
          store(blk, &raw.Hh[raw.pair_at(u, v, k)], cw.at(rows.Hh(q), k, g0));
          store(blk, &raw.corr[raw.pair_at(u, v, k)], cw.at(rows.corr(q), k, g0));
        }
      }
    }
    if (raw.ndh >= 0) {
      Weights& dw = h.dh_w[c];
      for (int k = dw.k0(); k < dw.end(); ++k) {
        for (int q = 0; q < dh_rows(nv); ++q) store(blk, &raw.Idh[raw.dh_at(q, k)], dw.at(q, k, g0));
      }
    }
    for (std::size_t q = 0; q < cls.keys.size(); ++q) {
      const int pat = static_cast<int>(q);
      Weights& kw = h.key_w[cls.keys[q]];
      for (int k = kw.k0(); k < kw.end(); ++k) {
        store(blk, &raw.chi[raw.pattern_at(pat, k)], kw.at(kChiRow, k, g0));
        store(blk, &raw.m0[raw.pattern_at(pat, k)], kw.at(kM0Row, k, g0));
      }
    }
  }

  static void subtract(const LaneBlock& blk, u64* dst, const u64* w) {
    for (int l = 0; l < blk.count; ++l) dst[l] = blk.mod[l].sub(dst[l], w[l]);
  }

  static void store(const LaneBlock& blk, const u64* src, u64* dst) { std::copy_n(src, blk.count, dst); }

  void check_vanishing(const LaneBlock& blk, const RawSeries& raw, int c, const LeadingOrders& lead) const {
    const int nv = raw.nv;
    auto zero = [&](const u64* x) {
      for (int l = 0; l < blk.count; ++l) {
        if (x[l]) return false;
      }
      return true;
    };
    bool ok = true;
    for (int k = 0; k < std::min(lead.energy, ng_ + 1) && ok; ++k) ok = zero(&raw.E[k * L]);
    for (int k = 0; k < std::min(lead.pair, ng_ + 1) && ok; ++k) {
      for (int u = 0; u < nv && ok; ++u) {
        for (int v = 0; v < nv && ok; ++v) {
          ok = zero(&raw.Hp[raw.pair_at(u, v, k)]) && zero(&raw.Hh[raw.pair_at(u, v, k)]) &&
               zero(&raw.corr[raw.pair_at(u, v, k)]);
        }
      }
    }
    for (int q = 0; q < raw.npat && ok; ++q) {
      for (int k = 0; k < std::min(lead.current, nc_ + 1) && ok; ++k) {
        ok = zero(&raw.chi[raw.pattern_at(q, k)]) && zero(&raw.m0[raw.pattern_at(q, k)]);
      }
    }
    if (raw.ndh >= 0) {
      const std::size_t rows = static_cast<std::size_t>(dh_rows(nv));
      for (std::size_t q = 0; q < rows && ok; ++q) {
        for (int k = 0; k < std::min(lead.dh, raw.ndh + 1) && ok; ++k) ok = zero(&raw.Idh[raw.dh_at(q, k)]);
      }
    }
    if (!ok) {
      throw std::logic_error("cluster cumulant does not vanish below its leading order (class " + std::to_string(c) +
                             ", " + std::to_string(nv) + " sites)");
    }
  }

  void accumulate(const LaneBlock& blk, std::size_t g0, const RawSeries& raw, const ClassInfo& cls,
                  const LeadingOrders& lead, const std::vector<DhEmbedding>& dh_emb,
                  std::vector<u64>& totals) const {
    const int k0 = lead.pair, k0c = lead.current;
    const std::size_t width = layout_.size();
    for (int l = 0; l < blk.count; ++l) {
      const Modulus& m = blk.mod[l];
      u64* tot = &totals[(g0 + l) * width];
      // Embeddings are counted per unit cell; the sums are per site.
      const u64 inv_cell = m.inv(m.from_int(geo_.lattice->cell_sites()));
      const u64 inv_order = m.mul(inv_cell, m.inv(m.from_int(static_cast<i64>(geo_.lattice->group.size()))));
      const u64 mult = m.mul(m.from_int(cls.mult), inv_cell);
      for (int k = k0; k <= ng_; ++k) tot[layout_.en() + k] = m.add(tot[layout_.en() + k], m.mul(mult, raw.E[k * L + l]));
      for (const Embedding& e : cls.embeddings) {
        const u64 f = m.mul(m.from_int(e.fac), inv_order);
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
      for (const Embedding& e : cls.pair_embeddings) {
        const u64 f = m.mul(m.from_int(e.fac), m.inv(m.from_int(geo_.pair_counts[e.cd])));
        const std::size_t off = static_cast<std::size_t>(e.cd) * (ng_ + 1);
        for (int k = k0; k <= ng_; ++k) {
          u64& hp = tot[layout_.hp_pairs() + off + k];
          hp = m.add(hp, m.mul(f, raw.Hp[raw.pair_at(e.a, e.b, k) + l]));
          u64& hh = tot[layout_.hh_pairs() + off + k];
          hh = m.add(hh, m.mul(f, raw.Hh[raw.pair_at(e.a, e.b, k) + l]));
        }
      }
      for (int q = 0; q < raw.npat; ++q) {
        const u64 km = m.mul(m.from_int(geo_.keys[cls.keys[q]].mult), inv_cell);
        const u64 km2 = m.add(km, km);
        for (int k = k0c; k <= nc_; ++k) {
          u64& ch = tot[layout_.chi() + k];
          ch = m.add(ch, m.mul(km2, raw.chi[raw.pattern_at(q, k) + l]));
          u64& mm = tot[layout_.m0() + k];
          mm = m.add(mm, m.mul(km, raw.m0[raw.pattern_at(q, k) + l]));
        }
      }
      const u64 inv_group = m.inv(m.from_int(static_cast<i64>(geo_.lattice->group.size())));
      for (const DhEmbedding& e : dh_emb) {
        const u64 f = m.mul(m.from_int(e.fac), inv_group);
        const std::size_t off = layout_.dh() + static_cast<std::size_t>(e.key) * (raw.ndh + 1);
        for (int k = lead.dh; k <= raw.ndh; ++k) {
          u64& x = tot[off + k];
          x = m.add(x, m.mul(f, raw.Idh[raw.dh_at(e.row, k) + l]));
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
