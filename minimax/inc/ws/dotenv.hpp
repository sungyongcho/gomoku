/* dotenv.hpp  –  trivial .env loader, C++98 */
#ifndef DOTENV_HPP
#define DOTENV_HPP
#include <cstdlib>
#include <fstream>
#include <string>

namespace dotenv {

inline void trim(std::string& s) {
  while (!s.empty() && (s[0] == ' ' || s[0] == '\t')) s.erase(0, 1);
  while (!s.empty() && (s[s.size() - 1] == ' ' || s[s.size() - 1] == '\t' ||
                        s[s.size() - 1] == '\r' || s[s.size() - 1] == '\n'))
    s.erase(s.size() - 1, 1);
}

inline bool init(const std::string& path = ".env") {
  std::ifstream f(path.c_str());
  if (!f.is_open()) return false;
  std::string line;
  while (std::getline(f, line)) {
    if (line.size() && line[0] != '#') {
      std::string::size_type eq = line.find('=');
      if (eq == std::string::npos) continue;
      std::string key = line.substr(0, eq);
      std::string val = line.substr(eq + 1);
      trim(key);
      trim(val);
      setenv(key.c_str(), val.c_str(), 1);
    }
  }
  return true;
}

int envToInt(const char* name, int dflt = 0) {
  const char* s = std::getenv(name);
  if (!s) return dflt;  // variable not set → use default

  char* end;
  long v = std::strtol(s, &end, 10);  // C-style, works in C++98
  if (*end != '\0')                   // not a pure number
    throw std::runtime_error(std::string(name) + " is not an integer");

  return static_cast<int>(v);  // safe here—ports fit in int
}

}  // namespace dotenv
#endif
