// Exact rationals from their residues modulo several primes.
#pragma once

#include "modp.hpp"

#include <string>
#include <vector>

namespace nlce {

struct Reconstruction {
  bool ok = false;
  std::vector<std::string> values;    // "p/q" or "p"; empty unless ok
  int max_bits = 0;                   // largest numerator + denominator bit length
  std::vector<std::size_t> unsolved;  // coefficients that failed, when not ok
};

// residues[i][c] is coefficient c modulo primes[i].  All but the last `ncheck`
// primes are combined by the Chinese remainder theorem and rationally
// reconstructed; the last `ncheck` primes must then reproduce every value.
// Every coefficient is attempted, so `unsolved` lists all that failed.
Reconstruction reconstruct(const std::vector<u64>& primes, const std::vector<std::vector<u64>>& residues,
                           int ncheck);

}  // namespace nlce
