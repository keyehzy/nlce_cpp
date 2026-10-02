#include "driver.hpp"

#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace nlce {

namespace {

double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void lattice_series(const Geometry& geo, int ng, int nc, const std::vector<Rational>& vs, const DriverOptions& opts,
                    const std::function<void(const LatticeSeries&)>& done) {
  const int ncheck = opts.check_primes;
  const int fixed = opts.fixed_primes;
  PassOptions pass_opts;
  pass_opts.threads = opts.threads;
  pass_opts.verbose = opts.verbose;

  struct State {
    Rational v;
    int target = 0;
    std::vector<u64> primes;
    std::vector<std::vector<u64>> residues;
    bool done = false;
  };
  std::vector<State> states;
  for (const auto& v : vs) states.push_back({v, fixed > 0 ? fixed + ncheck : 8 + ncheck, {}, {}, false});

  std::vector<u64> pool;
  for (int pass = 1;; ++pass) {
    // A partly filled lane block costs as much as a full one, so pad the pass
    // with further primes for the V/U that will need the most.
    int pending = 0;
    State* hungriest = nullptr;
    for (State& st : states) {
      if (st.done) continue;
      pending += st.target - static_cast<int>(st.primes.size());
      if (!hungriest || st.target > hungriest->target) hungriest = &st;
    }
    if (hungriest) hungriest->target += (kLanes - pending % kLanes) % kLanes;

    std::vector<LaneSpec> lanes;
    std::vector<std::pair<int, u64>> owner;
    for (int i = 0; i < static_cast<int>(states.size()); ++i) {
      State& st = states[i];
      if (st.done) continue;
      if (static_cast<int>(pool.size()) < st.target) pool = moduli(st.target);
      for (int k = static_cast<int>(st.primes.size()); k < st.target; ++k) {
        lanes.push_back({st.v, pool[k]});
        owner.emplace_back(i, pool[k]);
      }
    }
    if (lanes.empty()) break;
    const auto tp = std::chrono::steady_clock::now();
    if (opts.verbose) std::fprintf(stderr, "pass %d: %zu lanes, %d threads\n", pass, lanes.size(), opts.threads);
    auto res = run_pass(geo, ng, nc, lanes, pass_opts);
    if (opts.verbose) std::fprintf(stderr, "pass %d done in %.1fs\n", pass, since(tp));
    for (std::size_t g = 0; g < lanes.size(); ++g) {
      State& st = states[owner[g].first];
      st.primes.push_back(owner[g].second);
      st.residues.push_back(std::move(res[g]));
    }

    for (State& st : states) {
      if (st.done) continue;
      Reconstruction rec = reconstruct(st.primes, st.residues, ncheck);
      if (!rec.ok) {
        if (fixed > 0) throw std::runtime_error("reconstruction failed for V/U = " + to_string(st.v) + "; raise --primes");
        const int have = static_cast<int>(st.primes.size()) - ncheck;
        st.target = have + std::max(4, have / 2) + ncheck;
        if (opts.verbose) {
          std::fprintf(stderr, "  V/U=%s: %d primes insufficient, retrying with %d\n", to_string(st.v).c_str(), have,
                       st.target - ncheck);
        }
        continue;
      }
      st.done = true;
      done({st.v, static_cast<int>(st.primes.size()) - ncheck, std::move(rec)});
    }
  }
}

ClusterSeries cluster_series(const ClusterPlan& plan, const Rational& v, int check_primes) {
  const int ncheck = check_primes;
  std::vector<u64> primes;
  std::vector<std::vector<u64>> residues;
  for (int target = 8;; target += target / 2) {
    const auto pool = moduli(target + ncheck);
    RawSeries raw;
    for (std::size_t b = primes.size(); b < pool.size(); b += kLanes) {
      LaneBlock blk;
      blk.count = static_cast<int>(std::min<std::size_t>(kLanes, pool.size() - b));
      for (int l = 0; l < kLanes; ++l) {
        blk.mod[l] = Modulus(pool[b + std::min(l, blk.count - 1)]);
        blk.v[l] = v;
      }
      compute_block(plan, blk, raw);
      for (int l = 0; l < blk.count; ++l) {
        std::vector<u64> flat;
        auto take = [&](const std::vector<u64>& x) {
          for (std::size_t i = 0; i < x.size() / kLanes; ++i) flat.push_back(x[i * kLanes + l]);
        };
        take(raw.E);
        take(raw.Hp);
        take(raw.Hh);
        take(raw.corr);
        take(raw.chi);
        take(raw.m0);
        primes.push_back(pool[b + l]);
        residues.push_back(std::move(flat));
      }
    }
    Reconstruction rec = reconstruct(primes, residues, ncheck);
    if (rec.ok) return {static_cast<int>(primes.size()) - ncheck, std::move(rec)};
  }
}

}  // namespace nlce
