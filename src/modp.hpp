// Prime-field arithmetic for the multi-modular series computation.
//
// Every quantity in the linked-cluster pipeline is a polynomial expression in
// the hopping coefficients divided by unperturbed energy denominators, so the
// whole calculation can be carried out in Z/pZ and the exact rationals
// recovered afterwards by Chinese remaindering and rational reconstruction.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nlce {

using u64 = std::uint64_t;
using i64 = std::int64_t;
using u128 = unsigned __int128;

// Moduli are primes below 2^50.  The bound leaves headroom for lazy
// accumulation: a small integer (< 16) times a residue is < 2^54, so a sum of
// a few hundred of those still fits a u64, and a product of two residues is
// < 2^100, so a u128 absorbs 2^20 of them before it has to be folded.
inline constexpr int kPrimeBits = 50;

class Modulus {
public:
  Modulus() = default;
  explicit Modulus(u64 p);

  u64 p() const { return p_; }

  u64 reduce(u64 x) const {
    u64 q = static_cast<u64>((static_cast<u128>(x) * barrett_) >> 64);
    u64 r = x - q * p_;
    while (r >= p_) r -= p_;
    return r;
  }

  u64 reduce(u128 x) const {
    u64 h = reduce(static_cast<u64>(x >> 64));
    u64 l = reduce(static_cast<u64>(x));
    u64 t = mul(h, r64_) + l;
    return t >= p_ ? t - p_ : t;
  }

  // a, b < p.  The quotient is estimated in double precision; with p < 2^50
  // the estimate is off by at most two, which the signed correction absorbs.
  u64 mul(u64 a, u64 b) const {
    u64 q = static_cast<u64>(static_cast<double>(a) * static_cast<double>(b) * pinv_);
    i64 r = static_cast<i64>(a * b - q * p_);
    const i64 p = static_cast<i64>(p_);
    while (r < 0) r += p;
    while (r >= p) r -= p;
    return static_cast<u64>(r);
  }

  u64 add(u64 a, u64 b) const {
    u64 t = a + b;
    return t >= p_ ? t - p_ : t;
  }
  u64 sub(u64 a, u64 b) const { return a >= b ? a - b : a + p_ - b; }
  u64 neg(u64 a) const { return a ? p_ - a : 0; }
  u64 from_int(i64 x) const;
  u64 inv(u64 a) const;

private:
  u64 p_ = 0;
  u64 barrett_ = 0;  // floor(2^64 / p)
  u64 r64_ = 0;      // 2^64 mod p
  double pinv_ = 0.0;
};

// The first `count` primes below 2^kPrimeBits, in decreasing order.
std::vector<u64> moduli(std::size_t count);

// Lanes are independent (V/U, prime) evaluations that share every
// v-independent structure of a cluster.  A block of kLanes lanes is processed
// together so that one traversal of the sparse structure serves all of them.
inline constexpr int kLanes = 8;

struct Rational {
  i64 num = 0;
  i64 den = 1;
};

struct LaneBlock {
  int count = 0;  // live lanes; the remainder repeat the last live lane
  std::array<Modulus, kLanes> mod{};
  std::array<Rational, kLanes> v{};
  std::array<u64, kLanes> vres{};  // V/U mod p
};

}  // namespace nlce
