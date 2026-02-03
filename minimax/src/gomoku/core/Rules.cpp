#include "Rules.hpp"

#include <iostream>

#include "Board.hpp"
#include "Evaluation.hpp"
#include "ForbiddenPointFinder.h"


namespace {

// Bitmask-based capture check.
// Pattern: opponent at (x+dx,y+dy) and (x+2*dx,y+2*dy), current player at (x+3*dx,y+3*dy).
bool bitmaskCheckCapture(Board& board, int x, int y, int currentPlayer, int dx, int dy) {
  int opponent = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  uint64_t* P = board.getBitboardByPlayer(currentPlayer);
  uint64_t* O = board.getBitboardByPlayer(opponent);

  int cx1 = x + dx, cy1 = y + dy;
  int cx2 = x + 2 * dx, cy2 = y + 2 * dy;
  int cx3 = x + 3 * dx, cy3 = y + 3 * dy;
  if (!Board::isValidCoordinate(cx1, cy1) || !Board::isValidCoordinate(cx2, cy2) ||
      !Board::isValidCoordinate(cx3, cy3))
    return false;

  uint64_t mask1 = 1ULL << cx1;
  uint64_t mask2 = 1ULL << cx2;
  uint64_t mask3 = 1ULL << cx3;

  return (((O[cy1] & mask1) != 0) && ((O[cy2] & mask2) != 0) && ((P[cy3] & mask3) != 0));
}

inline void storeCapturedPair(Board& board, int x, int y, int dx, int dy, int player) {
  board.storeCapturedStone(x + dx, y + dy, OPPONENT(player));
  board.storeCapturedStone(x + 2 * dx, y + 2 * dy, OPPONENT(player));
}

bool detectCaptureStonesImpl(Board& board, int x, int y, int player, bool store) {
  bool foundCapture = false;
  for (size_t i = 0; i < 8; ++i) {
    int dx = DIRECTIONS[i][0];
    int dy = DIRECTIONS[i][1];
    if (bitmaskCheckCapture(board, x, y, player, dx, dy)) {
      if (store) {
        storeCapturedPair(board, x, y, dx, dy, player);
      } else {
        return true;
      }
      foundCapture = true;
    }
  }
  return foundCapture;
}

}  // namespace

bool Rules::detectCaptureStones(Board& board, int x, int y, int player) {
  return detectCaptureStonesImpl(board, x, y, player, true);
}

bool Rules::detectCaptureStonesNotStore(Board& board, int x, int y, int player) {
  return detectCaptureStonesImpl(board, x, y, player, false);
}

// Classical implementation using CForbiddenPointFinder
bool Rules::detectDoublethree(Board& board, int x, int y, int player) {
  CForbiddenPointFinder finder(BOARD_SIZE);

  for (int row = 0; row < BOARD_SIZE; ++row) {
    for (int col = 0; col < BOARD_SIZE; ++col) {
      int cell = board.getValueBit(col, row);
      if (cell == EMPTY_SPACE) continue;
      if (cell == player) {
        finder.SetStone(col, row, BLACKSTONE);
      } else {
        finder.SetStone(col, row, WHITESTONE);
      }
    }
  }

  return finder.IsDoubleThree(x, y);
}



bool Rules::isWinningMove(Board* board, int player, int x, int y) {
  return (MINIMAX_TERMINATION <= Evaluation::evaluatePositionHard(board, player, x, y));
}
