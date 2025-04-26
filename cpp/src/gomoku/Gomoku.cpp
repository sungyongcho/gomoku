#include "Gomoku.hpp"
namespace Zobrist {
ZobristKey piece_keys[BOARD_SIZE][BOARD_SIZE][3];
ZobristKey capture_keys[3][5 + 1];
ZobristKey turn_key;
// Global/static variables are zero-initialized by default if not specified
bool initialized = false;
}  // namespace Zobrist

boost::random::mt19937 global_boost_rng;

ZobristKey generate_random_64bit_boost() {
  // Combine two 32-bit outputs from mt19937 to form a 64-bit key
  ZobristKey high = static_cast<ZobristKey>(global_boost_rng());
  ZobristKey low = static_cast<ZobristKey>(global_boost_rng());
  ZobristKey key = (high << 32) | low;

  // Ensure non-zero key
  if (key == 0) {
    key = 1;
  }
  return key;
}

void initZobrist() {
  if (Zobrist::initialized) {
    return;  // Already initialized
  }

  // Seed the global Boost generator ONCE
  // Using time(0) is the standard C++98 compatible way.
  // boost::random::random_device could be used if available/preferred for better seeding,
  // but requires checking its availability and usage. time(0) is simpler.
  unsigned int seed = static_cast<unsigned int>(std::time(0));
  global_boost_rng.seed(seed);

  // Generate keys using the helper function
  for (int x = 0; x < BOARD_SIZE; ++x) {
    for (int y = 0; y < BOARD_SIZE; ++y) {
      Zobrist::piece_keys[x][y][PLAYER_1] = generate_random_64bit_boost();
      Zobrist::piece_keys[x][y][PLAYER_2] = generate_random_64bit_boost();
    }
  }

  for (int p = 1; p <= 2; p++) {
    for (int points = 0; points <= 5; ++points) {
      Zobrist::capture_keys[p][points] = generate_random_64bit_boost();
    }
  }

  Zobrist::turn_key = generate_random_64bit_boost();

  Zobrist::initialized = true;
  std::cout << "Zobrist keys initialized successfully using Boost.Random (mt19937)." << std::endl;
}
