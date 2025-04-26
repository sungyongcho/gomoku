#ifndef RULES_HPP
#define RULES_HPP

#include <vector>

#include "Evaluation.hpp"
#include "Gomoku.hpp"

class Board;
class Rules {
 public:
  static bool detectCaptureStones(Board &board, int x, int y, int player);
  static bool detectCaptureStonesNotStore(Board &board, int x, int y, int player);

  static bool detectDoublethreeBit(Board &board, int x, int y, int player);
  static bool isWinningMove(Board *board, int player, int x, int y);
};

#endif  // RULES_HPP
