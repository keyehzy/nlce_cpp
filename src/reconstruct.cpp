#include "reconstruct.hpp"

#include <flint/flint.h>
#include <flint/fmpq.h>
#include <flint/fmpz.h>

#include <algorithm>
#include <stdexcept>

namespace nlce {

namespace {

struct Fmpz {
  fmpz_t x;
  Fmpz() { fmpz_init(x); }
  ~Fmpz() { fmpz_clear(x); }
  Fmpz(const Fmpz&) = delete;
  Fmpz& operator=(const Fmpz&) = delete;
};

struct Fmpq {
  fmpq_t x;
  Fmpq() { fmpq_init(x); }
  ~Fmpq() { fmpq_clear(x); }
  Fmpq(const Fmpq&) = delete;
  Fmpq& operator=(const Fmpq&) = delete;
};

}  // namespace

Reconstruction reconstruct(const std::vector<u64>& primes, const std::vector<std::vector<u64>>& residues,
                           int ncheck) {
  if (primes.size() != residues.size()) throw std::invalid_argument("one residue vector per prime");
  const int nrec = static_cast<int>(primes.size()) - ncheck;
  Reconstruction out;
  if (nrec < 1) return out;
  const std::size_t ncoef = residues[0].size();

  std::vector<Fmpz> moduli(nrec);
  fmpz_set_ui(moduli[0].x, primes[0]);
  for (int i = 1; i < nrec; ++i) fmpz_mul_ui(moduli[i].x, moduli[i - 1].x, primes[i]);

  Fmpz a, b;
  Fmpq q;
  out.values.reserve(ncoef);
  for (std::size_t c = 0; c < ncoef; ++c) {
    fmpz_set_ui(a.x, residues[0][c]);
    for (int i = 1; i < nrec; ++i) {
      fmpz_CRT_ui(b.x, a.x, moduli[i - 1].x, residues[i][c], primes[i], 0);
      fmpz_swap(a.x, b.x);
    }
    bool solved = fmpq_reconstruct_fmpz(q.x, a.x, moduli[nrec - 1].x);
    for (int i = nrec; i < static_cast<int>(primes.size()) && solved; ++i) {
      const Modulus m(primes[i]);
      const u64 num = fmpz_fdiv_ui(fmpq_numref(q.x), primes[i]);
      const u64 den = fmpz_fdiv_ui(fmpq_denref(q.x), primes[i]);
      solved = den != 0 && m.mul(num, m.inv(den)) == residues[i][c];
    }
    if (!solved) {
      out.unsolved.push_back(c);
      continue;
    }
    if (!out.unsolved.empty()) continue;
    const int bits = static_cast<int>(fmpz_bits(fmpq_numref(q.x)) + fmpz_bits(fmpq_denref(q.x)));
    out.max_bits = std::max(out.max_bits, bits);
    char* s = fmpq_get_str(nullptr, 10, q.x);
    out.values.emplace_back(s);
    flint_free(s);
  }
  out.ok = out.unsolved.empty();
  if (!out.ok) {
    out.values.clear();
    out.max_bits = 0;
  }
  return out;
}

}  // namespace nlce
