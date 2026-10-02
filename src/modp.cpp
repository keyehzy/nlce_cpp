#include "modp.hpp"

#include <flint/ulong_extras.h>

#include <stdexcept>

namespace nlce {

Modulus::Modulus(u64 p) : p_(p) {
  if (p < 3 || p >= (u64(1) << kPrimeBits)) throw std::invalid_argument("modulus out of range");
  const u128 two64 = static_cast<u128>(1) << 64;
  barrett_ = static_cast<u64>(two64 / p);
  r64_ = static_cast<u64>(two64 % p);
  pinv_ = 1.0 / static_cast<double>(p);
}

u64 Modulus::from_int(i64 x) const {
  const i64 p = static_cast<i64>(p_);
  i64 r = x % p;
  if (r < 0) r += p;
  return static_cast<u64>(r);
}

u64 Modulus::inv(u64 a) const {
  if (a == 0) throw std::domain_error("inverse of zero");
  i64 r0 = static_cast<i64>(p_), r1 = static_cast<i64>(a);
  i64 s0 = 0, s1 = 1;
  while (r1) {
    i64 q = r0 / r1;
    i64 t = r0 - q * r1;
    r0 = r1;
    r1 = t;
    t = s0 - q * s1;
    s0 = s1;
    s1 = t;
  }
  return from_int(s0);
}

std::vector<u64> moduli(std::size_t count) {
  std::vector<u64> out;
  out.reserve(count);
  u64 n = (u64(1) << kPrimeBits) - 1;
  while (out.size() < count) {
    if (n_is_prime(n)) out.push_back(n);
    n -= 2;
  }
  return out;
}

}  // namespace nlce
