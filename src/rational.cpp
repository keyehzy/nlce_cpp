#include "rational.hpp"

#include <numeric>
#include <stdexcept>

namespace nlce {

namespace {

std::int64_t parse_integer(const std::string& s, const std::string& whole) {
  std::size_t used = 0;
  std::int64_t x = 0;
  try {
    x = std::stoll(s, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used == 0 || used != s.size()) throw std::invalid_argument("bad rational " + whole);
  return x;
}

}  // namespace

Rational parse_rational(const std::string& s) {
  const auto slash = s.find('/');
  Rational r;
  r.num = parse_integer(s.substr(0, slash), s);
  r.den = slash == std::string::npos ? 1 : parse_integer(s.substr(slash + 1), s);
  if (r.den <= 0) throw std::invalid_argument("bad rational " + s);
  const std::int64_t g = std::gcd(r.num, r.den);
  r.num /= g;
  r.den /= g;
  return r;
}

std::string to_string(const Rational& r) {
  return r.den == 1 ? std::to_string(r.num) : std::to_string(r.num) + "/" + std::to_string(r.den);
}

std::string file_tag(const Rational& r) {
  return r.den == 1 ? std::to_string(r.num) : std::to_string(r.num) + "over" + std::to_string(r.den);
}

}  // namespace nlce
