// nlce_run: exact site-cluster series for the frustrated triangular-lattice
// extended Bose-Hubbard model at unit filling.
//
//   nlce_run geometry --nsites N
//   nlce_run run --nsites N --v 0,1/20,1/4 --out DIR [--threads T] [--primes K]
//   nlce_run cluster --nv N --edges 0-1,1-2 --ng G --nc C --v 1/5 [--pattern +-] ...
//     ("none" stands for an empty edge list or pattern)
//   nlce_run bench --nv N --edges ... --ng G --nc C [--pattern ...] [--v 1/5] [--reps 3]
//
// `run` writes one JSON file per V/U with every series coefficient as an exact
// rational; tools/to_pickles.py converts them to the series/*.pkl format.
#include "cluster.hpp"
#include "geometry.hpp"
#include "pipeline.hpp"
#include "reconstruct.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace nlce;

namespace {

struct Args {
  std::string command;
  std::multimap<std::string, std::string> opts;

  bool has(const std::string& k) const { return opts.count(k) > 0; }
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = opts.find(k);
    return it == opts.end() ? def : it->second;
  }
  std::vector<std::string> all(const std::string& k) const {
    std::vector<std::string> out;
    auto [b, e] = opts.equal_range(k);
    for (auto it = b; it != e; ++it) out.push_back(it->second);
    return out;
  }
  int integer(const std::string& k, int def) const { return has(k) ? std::stoi(get(k)) : def; }
};

Args parse(int argc, char** argv) {
  if (argc < 2) throw std::invalid_argument("usage: nlce_run {geometry|run|cluster} [options]");
  Args a;
  a.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::invalid_argument("bad option " + k);
    a.opts.emplace(k.substr(2), argv[++i]);
  }
  return a;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

Rational parse_rational(const std::string& s) {
  const auto slash = s.find('/');
  Rational r;
  r.num = std::stoll(s.substr(0, slash));
  r.den = slash == std::string::npos ? 1 : std::stoll(s.substr(slash + 1));
  if (r.den <= 0) throw std::invalid_argument("bad rational " + s);
  i64 a = std::llabs(r.num), b = r.den;
  while (b) {
    i64 t = a % b;
    a = b;
    b = t;
  }
  if (a > 1) {
    r.num /= a;
    r.den /= a;
  }
  return r;
}

std::string rational_string(const Rational& r) {
  return r.den == 1 ? std::to_string(r.num) : std::to_string(r.num) + "/" + std::to_string(r.den);
}

// File tag matching the Python drivers: 1/20 -> "1over20".
std::string rational_tag(const Rational& r) {
  return r.den == 1 ? std::to_string(r.num) : std::to_string(r.num) + "over" + std::to_string(r.den);
}

double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void json_list(std::ostream& os, const std::vector<std::string>& v, std::size_t begin, std::size_t n) {
  os << '[';
  for (std::size_t i = 0; i < n; ++i) os << (i ? "," : "") << '"' << v[begin + i] << '"';
  os << ']';
}

int cmd_geometry(const Args& args) {
  const int nsites = args.integer("nsites", 0);
  const auto t0 = std::chrono::steady_clock::now();
  const Geometry geo = build_geometry(nsites);
  std::printf("geometry s<=%d built in %.2fs\n", nsites, since(t0));
  std::printf("%4s %10s %8s %8s\n", "s", "clusters", "classes", "keys");
  for (int s = 1; s <= nsites; ++s) {
    int classes = 0, keys = 0;
    for (const auto& c : geo.classes) classes += c.nv == s;
    for (const auto& k : geo.keys) keys += geo.classes[k.cls].nv == s;
    std::printf("%4d %10d %8d %8d\n", s, geo.cluster_counts[s], classes, keys);
  }
  std::printf("displacements %zu\n", geo.displacements.size());
  return 0;
}

int cmd_run(const Args& args) {
  const int nsites = args.integer("nsites", 0);
  if (nsites < 2) throw std::invalid_argument("--nsites must be at least 2");
  std::vector<Rational> vs;
  for (const auto& s : split(args.get("v", "0,1/20,1/10,3/20,1/5,1/4"), ',')) vs.push_back(parse_rational(s));
  const std::filesystem::path out_dir = args.get("out", ".");
  std::filesystem::create_directories(out_dir);
  const int fixed = args.integer("primes", 0);
  const int ncheck = args.integer("check", 2);
  PassOptions opts;
  opts.threads = args.integer("threads", static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
  const int ng = nsites - 1, nc = nsites - 2;

  const auto t0 = std::chrono::steady_clock::now();
  const Geometry geo = build_geometry(nsites);
  std::fprintf(stderr, "geometry: %zu classes, %zu keys, %zu displacements (%.1fs)\n", geo.classes.size(),
               geo.keys.size(), geo.displacements.size(), since(t0));

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
    std::fprintf(stderr, "pass %d: %zu lanes, %d threads\n", pass, lanes.size(), opts.threads);
    auto res = run_pass(geo, ng, nc, lanes, opts);
    std::fprintf(stderr, "pass %d done in %.1fs\n", pass, since(tp));
    for (std::size_t g = 0; g < lanes.size(); ++g) {
      State& st = states[owner[g].first];
      st.primes.push_back(owner[g].second);
      st.residues.push_back(std::move(res[g]));
    }

    const SeriesLayout lay{ng, nc, static_cast<int>(geo.displacements.size())};
    for (State& st : states) {
      if (st.done) continue;
      const Reconstruction rec = reconstruct(st.primes, st.residues, ncheck);
      if (!rec.ok) {
        if (fixed > 0) throw std::runtime_error("reconstruction failed for V/U = " + rational_string(st.v) + "; raise --primes");
        const int have = static_cast<int>(st.primes.size()) - ncheck;
        st.target = have + std::max(4, have / 2) + ncheck;
        std::fprintf(stderr, "  V/U=%s: %d primes insufficient, retrying with %d\n", rational_string(st.v).c_str(),
                     have, st.target - ncheck);
        continue;
      }
      st.done = true;
      const auto path = out_dir / ("series_s" + std::to_string(nsites) + "_v" + rational_tag(st.v) + ".json");
      std::ofstream os(path);
      os << "{\n\"nsites\": " << nsites << ",\n\"v\": \"" << rational_string(st.v) << "\",\n\"order_gap\": " << ng
         << ",\n\"order_chi\": " << nc << ",\n\"primes\": " << st.primes.size() - ncheck
         << ",\n\"check_primes\": " << ncheck << ",\n\"max_bits\": " << rec.max_bits << ",\n\"displacements\": [";
      for (std::size_t d = 0; d < geo.displacements.size(); ++d) {
        os << (d ? "," : "") << '[' << geo.displacements[d].a << ',' << geo.displacements[d].b << ']';
      }
      os << "],\n\"EN\": ";
      json_list(os, rec.values, lay.en(), ng + 1);
      for (auto [name, off] : {std::pair{"Hp", lay.hp()}, std::pair{"Hh", lay.hh()}, std::pair{"S", lay.s()}}) {
        os << ",\n\"" << name << "\": [";
        for (int d = 0; d < lay.ncd; ++d) {
          os << (d ? ",\n  " : "");
          json_list(os, rec.values, off + static_cast<std::size_t>(d) * (ng + 1), ng + 1);
        }
        os << ']';
      }
      os << ",\n\"chi\": ";
      json_list(os, rec.values, lay.chi(), nc + 1);
      os << ",\n\"m0\": ";
      json_list(os, rec.values, lay.m0(), nc + 1);
      os << "\n}\n";
      std::fprintf(stderr, "  V/U=%s: reconstructed from %zu primes (%d bits) -> %s\n", rational_string(st.v).c_str(),
                   st.primes.size() - ncheck, rec.max_bits, path.string().c_str());
    }
  }
  std::fprintf(stderr, "total %.1fs\n", since(t0));
  return 0;
}

int cmd_cluster(const Args& args) {
  ClusterInput in;
  in.nv = args.integer("nv", 0);
  in.ng = args.integer("ng", 0);
  in.nc = args.integer("nc", 0);
  for (const auto& e : split(args.get("edges", "none"), ',')) {
    if (e == "none") continue;
    const auto parts = split(e, '-');
    in.edges.emplace_back(std::stoi(parts.at(0)), std::stoi(parts.at(1)));
  }
  for (const auto& p : args.all("pattern")) {
    Pattern pat;
    if (p != "none")
      for (char ch : p) pat.push_back(static_cast<std::int8_t>(ch == '+' ? 1 : -1));
    in.patterns.push_back(pat);
  }
  const Rational v = parse_rational(args.get("v", "0"));
  const int ncheck = 2;
  const ClusterPlan plan = plan_cluster(in);
  const int nv = in.nv, ng = in.ng, nc = in.nc, npat = static_cast<int>(in.patterns.size());

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
    const Reconstruction rec = reconstruct(primes, residues, ncheck);
    if (!rec.ok) continue;
    std::size_t at = 0;
    auto emit = [&](const char* name, int blocks, int len) {
      std::printf("%s\"%s\": [", at ? ",\n" : "{\n", name);
      for (int q = 0; q < blocks; ++q) {
        std::printf("%s[", q ? "," : "");
        for (int k = 0; k < len; ++k) std::printf("%s\"%s\"", k ? "," : "", rec.values[at++].c_str());
        std::printf("]");
      }
      std::printf("]");
    };
    emit("E", 1, ng + 1);
    emit("Hp", nv * nv, ng + 1);
    emit("Hh", nv * nv, ng + 1);
    emit("corr", nv * nv, ng + 1);
    emit("chi", npat, nc + 1);
    emit("m0", npat, nc + 1);
    std::printf(",\n\"primes\": %zu\n}\n", primes.size() - ncheck);
    return 0;
  }
}

// Times the plan and one lane block of a single cluster.
int cmd_bench(const Args& args) {
  ClusterInput in;
  in.nv = args.integer("nv", 0);
  in.ng = args.integer("ng", 0);
  in.nc = args.integer("nc", 0);
  for (const auto& e : split(args.get("edges", "none"), ',')) {
    if (e == "none") continue;
    const auto parts = split(e, '-');
    in.edges.emplace_back(std::stoi(parts.at(0)), std::stoi(parts.at(1)));
  }
  for (const auto& p : args.all("pattern")) {
    Pattern pat;
    for (char ch : p) pat.push_back(static_cast<std::int8_t>(ch == '+' ? 1 : -1));
    in.patterns.push_back(pat);
  }
  const int reps = args.integer("reps", 3);
  auto t0 = std::chrono::steady_clock::now();
  const ClusterPlan plan = plan_cluster(in);
  std::printf("plan %.3fs  mott %u  particle %u  hole %u  pair moves %zu\n", since(t0), plan.mott.size(),
              plan.particle.size(), plan.hole.size(), plan.pairs.size());
  LaneBlock blk;
  blk.count = kLanes;
  const auto pool = moduli(kLanes);
  for (int l = 0; l < kLanes; ++l) {
    blk.mod[l] = Modulus(pool[l]);
    blk.v[l] = parse_rational(args.get("v", "1/5"));
  }
  RawSeries raw;
  for (int r = 0; r < reps; ++r) {
    t0 = std::chrono::steady_clock::now();
    compute_block(plan, blk, raw);
    std::printf("block %.3fs\n", since(t0));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    if (args.command == "geometry") return cmd_geometry(args);
    if (args.command == "run") return cmd_run(args);
    if (args.command == "cluster") return cmd_cluster(args);
    if (args.command == "bench") return cmd_bench(args);
    throw std::invalid_argument("unknown command " + args.command);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
