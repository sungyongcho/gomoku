#include "Gomoku.hpp"

#include <cstdlib>
#include <ctime>

uint64_t zobristTable[NUM_CELLS][3];
uint64_t zobristTurn[3];

void initZobrist() {
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < NUM_CELLS; ++i) {
    for (int state = 0; state < 3; ++state) {
      zobristTable[i][state] = ((uint64_t)rand() << 32) | rand();
    }
  }
  zobristTurn[1] = ((uint64_t)rand() << 32) | rand();
  zobristTurn[2] = ((uint64_t)rand() << 32) | rand();
}
