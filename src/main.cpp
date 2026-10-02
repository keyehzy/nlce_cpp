// nlce_run: exact site-cluster series for the extended Bose-Hubbard model at
// unit filling, H = x sum_<ij> b^dag_i b_j + h.c. + (1/2) sum_i n_i(n_i - 1)
// + (V/U) sum_<ij> (n_i - 1)(n_j - 1), on the chain, square or (frustrated)
// triangular lattice.
//
//   nlce_run geometry --nsites N [--lattice L]
//   nlce_run run --nsites N --v 0,1/20,1/4 --out DIR [--lattice L] [--threads T] [--primes K]
//   nlce_run cluster --nv N --edges 0-1,1-2 --ng G --nc C --v 1/5 [--pattern +-] ...
//     ("none" stands for an empty edge list or pattern)
//   nlce_run bench --nv N --edges ... --ng G --nc C [--pattern ...] [--v 1/5] [--reps 3]
//
// The lattice is chain, square or triangular (the default).  `run` writes one
// JSON file per V/U with every series coefficient as an exact rational;
// tools/to_pickles.py converts them to the series/*.pkl format.  chi and m0
// are computed on the triangular lattice only.
#include "cluster.hpp"
#include "driver.hpp"
#include "geometry.hpp"
#include "output.hpp"
#include "rational.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
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
  if (argc < 2) throw std::invalid_argument("usage: nlce_run {geometry|run|cluster|bench} [options]");
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

double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// --nv, --ng, --nc, --edges and repeated --pattern, shared by `cluster` and
// `bench`.
ClusterInput parse_cluster_input(const Args& args) {
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
  return in;
}

const Lattice& parse_lattice(const Args& args) { return lattice_by_name(args.get("lattice", "triangular")); }

int cmd_geometry(const Args& args) {
  const int nsites = args.integer("nsites", 0);
  const Lattice& lat = parse_lattice(args);
  const auto t0 = std::chrono::steady_clock::now();
  const Geometry geo = build_geometry(lat, nsites);
  std::printf("%s geometry s<=%d built in %.2fs\n", lat.name.c_str(), nsites, since(t0));
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
  const Lattice& lat = parse_lattice(args);
  std::vector<Rational> vs;
  for (const auto& s : split(args.get("v", "0,1/20,1/10,3/20,1/5,1/4"), ',')) vs.push_back(parse_rational(s));
  const std::filesystem::path out_dir = args.get("out", ".");
  std::filesystem::create_directories(out_dir);
  DriverOptions opts;
  opts.fixed_primes = args.integer("primes", 0);
  opts.check_primes = args.integer("check", 2);
  opts.threads = args.integer("threads", static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
  const int ng = nsites - 1, nc = lat.currents ? nsites - 2 : 0;

  const auto t0 = std::chrono::steady_clock::now();
  const Geometry geo = build_geometry(lat, nsites);
  std::fprintf(stderr, "%s geometry: %zu classes, %zu keys, %zu displacements (%.1fs)\n", lat.name.c_str(),
               geo.classes.size(), geo.keys.size(), geo.displacements.size(), since(t0));

  lattice_series(geo, ng, nc, vs, opts, [&](const LatticeSeries& series) {
    const auto path = out_dir / ("series_s" + std::to_string(nsites) + "_v" + file_tag(series.v) + ".json");
    std::ofstream os(path);
    write_lattice_json(os, lat, nsites, ng, nc, geo.displacements, series, opts.check_primes);
    std::fprintf(stderr, "  V/U=%s: reconstructed from %d primes (%d bits) -> %s\n", to_string(series.v).c_str(),
                 series.primes, series.rec.max_bits, path.string().c_str());
  });
  std::fprintf(stderr, "total %.1fs\n", since(t0));
  return 0;
}

int cmd_cluster(const Args& args) {
  const ClusterPlan plan = plan_cluster(parse_cluster_input(args));
  const ClusterSeries series = cluster_series(plan, parse_rational(args.get("v", "0")), 2);
  const ClusterInput& in = plan.in;
  write_cluster_json(std::cout, in.nv, in.ng, in.nc, static_cast<int>(in.patterns.size()), series);
  return 0;
}

// Times the plan and one lane block of a single cluster.
int cmd_bench(const Args& args) {
  const ClusterInput in = parse_cluster_input(args);
  const int reps = args.integer("reps", 3);
  auto t0 = std::chrono::steady_clock::now();
  const ClusterPlan plan = plan_cluster(in);
  std::printf("plan %.3fs  mott %u  particle %u  hole %u  pair moves %zu\n", since(t0), plan.mott.size(),
              plan.particle.size(), plan.hole.size(), plan.pairs.size());
  std::vector<LaneSpec> specs;
  for (u64 p : moduli(kLanes)) specs.push_back({parse_rational(args.get("v", "1/5")), p});
  const LaneBlock blk = make_lane_block(specs, 0);
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
