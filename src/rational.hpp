// Small exact rationals for the model parameter V/U.
#pragma once

#include <cstdint>
#include <string>

namespace nlce {

struct Rational {
  std::int64_t num = 0;
  std::int64_t den = 1;
  bool operator==(const Rational&) const = default;
};

// "p" or "p/q" with q > 0, reduced to lowest terms.
Rational parse_rational(const std::string& s);

// "p/q", or "p" when q = 1.
std::string to_string(const Rational& r);

// File-name tag used by the Python drivers: 1/20 -> "1over20".
std::string file_tag(const Rational& r);

}  // namespace nlce
