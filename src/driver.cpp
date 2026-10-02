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

// PRIME SCHEDULE.  Residues from every pass are kept, so a further pass costs
// only its replanning, a small fraction of one lane block, while every prime
// beyond what a V/U needs costs a full lane.  So each V/U starts low and grows
// by a fifth.  The spare lanes that fill a pass's last block go to the V/U with
// the most orders still unsolved.  At s = 9 and 10 this lands within a block or
// so of the fewest lanes that could reconstruct the series.
constexpr int kInitialPrimes = 3;

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
    int unsolved_orders = 0;  // orders with a coefficient the last attempt missed
    bool done = false;
  };
  std::vector<State> states;
  for (const auto& v : vs) states.push_back({v, (fixed > 0 ? fixed : kInitialPrimes) + ncheck, {}, {}, 0, false});
  const SeriesLayout layout{ng, nc, static_cast<int>(geo.displacements.size())};

  std::vector<u64> pool;
  for (int pass = 1;; ++pass) {
    // A partly filled lane block costs as much as a full one, so pad the pass
    // with further primes for the V/U that will need the most.
    int pending = 0;
    State* hungriest = nullptr;
    for (State& st : states) {
      if (st.done) continue;
      pending += st.target - static_cast<int>(st.primes.size());
      if (!hungriest || std::pair(st.unsolved_orders, st.target) > std::pair(hungriest->unsolved_orders, hungriest->target)) {
        hungriest = &st;
      }
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
        st.target = have + std::max(4, (have + 4) / 5) + ncheck;
        int lowest = ng;
        for (std::size_t c : rec.unsolved) lowest = std::min(lowest, layout.order(c));
        st.unsolved_orders = ng + 1 - lowest;
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
    std::vector<LaneSpec> specs;
    for (u64 p : moduli(target + ncheck)) specs.push_back({v, p});
    RawSeries raw;
    for (std::size_t b = primes.size(); b < specs.size(); b += kLanes) {
      const LaneBlock blk = make_lane_block(specs, b);
      compute_block(plan, blk, raw);
      for (int l = 0; l < blk.count; ++l) {
        primes.push_back(specs[b + l].p);
        residues.push_back(raw.lane(l));
      }
    }
    Reconstruction rec = reconstruct(primes, residues, ncheck);
    if (rec.ok) return {static_cast<int>(primes.size()) - ncheck, std::move(rec)};
  }
}

}  // namespace nlce
