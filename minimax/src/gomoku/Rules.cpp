#include "Rules.hpp"

#include <iostream>

#include "Board.hpp"
#include "ForbiddenPointFinder.h"
#include "ForbiddenPointFinderBit.h"

#ifndef DOUBLETHREE_VERIFY
#define DOUBLETHREE_VERIFY 0
#endif

// Bitmask-based capture check that stores captured stone coordinates.
// It checks for the pattern: opponent stone at (x+dx, y+dy) and (x+2*dx,
// y+2*dy), with currentPlayer’s stone at (x+3*dx, y+3*dy). If the pattern is
// found, it removes the captured stones from the opponent's bitboard and adds
// their coordinates to the captured vector.
bool bitmask_check_capture(Board& board, int x, int y, int currentPlayer, int dx, int dy) {
  int opponent = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  uint64_t* P = board.getBitboardByPlayer(currentPlayer);
  uint64_t* O = board.getBitboardByPlayer(opponent);

  int cx1 = x + dx, cy1 = y + dy;
  int cx2 = x + 2 * dx, cy2 = y + 2 * dy;
  int cx3 = x + 3 * dx, cy3 = y + 3 * dy;
  if (!Board::isValidCoordinate(cx1, cy1) || !Board::isValidCoordinate(cx2, cy2) ||
      !Board::isValidCoordinate(cx3, cy3))
    return false;

  // In the new representation, each row is stored in one uint64_t.
  // Compute the mask for each cell: the bit at position 'col' in that row.
  uint64_t mask1 = 1ULL << cx1;
  uint64_t mask2 = 1ULL << cx2;
  uint64_t mask3 = 1ULL << cx3;

  // The pattern: cell at (cx1, cy1) and (cx2, cy2) must be occupied by the
  // opponent, and the cell at (cx3, cy3) must be occupied by the current
  // player.
  return (((O[cy1] & mask1) != 0) && ((O[cy2] & mask2) != 0) && ((P[cy3] & mask3) != 0));
}

bool Rules::detectCaptureStones(Board& board, int x, int y, int player) {
  bool foundCapture = false;
  // Loop over the 8 directions.
  for (size_t i = 0; i < 8; ++i) {
    int dx = DIRECTIONS[i][0];
    int dy = DIRECTIONS[i][1];
    if (bitmask_check_capture(board, x, y, player, dx, dy)) {
      // Captured stones are at (x+dx, y+dy) and (x+2*dx, y+2*dy).
      board.storeCapturedStone(x + dx, y + dy, OPPONENT(player));
      board.storeCapturedStone(x + 2 * dx, y + 2 * dy, OPPONENT(player));
      foundCapture = true;
    }
  }
  return foundCapture;
}

bool Rules::detectCaptureStonesNotStore(Board& board, int x, int y, int player) {
  // Loop over the 8 directions.
  for (size_t i = 0; i < 8; ++i) {
    int dx = DIRECTIONS[i][0];
    int dy = DIRECTIONS[i][1];
    if (bitmask_check_capture(board, x, y, player, dx, dy)) {
      return true;
    }
  }
  return false;
}

bool detectDoublethreeRenju(const Board& board, int x, int y, int player) {
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

// returns true if placing a stone at (x,y) for 'player' creates a double-three
// (bitwise).
bool Rules::detectDoublethreeBit(Board& board, int x, int y, int player) {
  const bool bit = ForbiddenPointFinderBit::IsDoubleThree(board, x, y, player);

#if DOUBLETHREE_VERIFY
  const bool ref = detectDoublethreeRenju(board, x, y, player);
  if (bit != ref) {
    std::cerr << "[doublethree mismatch] x=" << x << " y=" << y << " player=" << player
              << " bit=" << bit << " ref=" << ref << std::endl;
  }
#endif

  // Debug swap: return detectDoublethreeRenju(board, x, y, player);
  return bit;
}

bool Rules::isWinningMove(Board* board, int player, int x, int y) {
  if (MINIMAX_TERMINATION <= Evaluation::evaluatePositionHard(board, player, x, y)) return true;
  return false;
}
