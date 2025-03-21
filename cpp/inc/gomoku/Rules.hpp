#ifndef RULES_HPP
#define RULES_HPP

#include <vector>

#include "Board.hpp"
#include "Gomoku.hpp"
#include "minimax.hpp"

class Rules {
 public:
  static bool detectCaptureStones(Board &board, int x, int y, const std::string &last_player);

  static bool detectDoublethreeBit(Board &board, int x, int y, int player);
  static bool isWinningMove(Board *board, int player, int x, int y);
};

#endif  // RULES_HPP
