#include "Rules.hpp"

#include <iostream>

// Bitmask-based capture check that stores captured stone coordinates.
// It checks for the pattern: opponent stone at (x+dx, y+dy) and (x+2*dx,
// y+2*dy), with currentPlayer’s stone at (x+3*dx, y+3*dy). If the pattern is
// found, it removes the captured stones from the opponent's bitboard and adds
// their coordinates to the captured vector.
bool bitmask_check_capture(Board &board, int x, int y, int currentPlayer, int dx, int dy) {
  int opponent = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  uint64_t *P = board.getBitboardByPlayer(currentPlayer);
  uint64_t *O = board.getBitboardByPlayer(opponent);

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

bool Rules::detectCaptureStones(Board &board, int x, int y, int player) {
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

bool Rules::detectCaptureStonesNotStore(Board &board, int x, int y, int player) {
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

bool is_within_bounds(int x, int y, int offset_x, int offset_y) {
  int new_x = x + offset_x;
  int new_y = y + offset_y;
  return (new_x >= 0 && new_x < BOARD_SIZE && new_y >= 0 && new_y < BOARD_SIZE);
}

// /**
//  * Helper functions to pack cell values into an unsigned int.
//  * Each cell uses 2 bits.
//  */
// inline unsigned int pack_cells_4(unsigned int a, unsigned int b, unsigned int c, unsigned int d)
// {
//   return (a << 6) | (b << 4) | (c << 2) | d;
// }

// inline unsigned int pack_cells_3(unsigned int a, unsigned int b, unsigned int c) {
//   return (a << 4) | (b << 2) | c;
// }

// inline unsigned int pack_cells_2(unsigned int a, unsigned int b) { return (a << 2) | b; }

// inline unsigned int pack_cells_1(unsigned int a) { return a; }

bool check_edge_bit_case_1(unsigned int forward, unsigned int backward, int player, int opponent) {
  unsigned int cond_1a_fwd = pack_cells_4(player, player, EMPTY_SPACE, opponent);
  unsigned int cond_1a_bkwd = pack_cells_2(EMPTY_SPACE, opponent);

  unsigned int cond_1b_fwd = pack_cells_4(player, player, EMPTY_SPACE, player);

  // Fallback expected pattern using only the first 3 forward cells
  // and the first backward cell.
  unsigned int cond1_fwd = pack_cells_3(player, player, EMPTY_SPACE);
  // unsigned int cond1_bkwd = pack_cells_1(EMPTY_SPACE);

  unsigned int forward_three_cells = (forward >> 2) & 0x3F;
  // unsigned int backward_one_cell = (backward >> 2) & 0x03;

  bool cond_1a = (cond_1a_fwd == forward) && (cond_1a_bkwd == backward);
  bool cond_1b = (cond_1b_fwd == forward);

  // if (!(cond_1a || cond_1b) && (cond1_fwd == forward_three_cells) &&
  //     (cond1_bkwd == backward_one_cell))
  //   return true;
  if (!(cond_1a || cond_1b) && (cond1_fwd == forward_three_cells)) return true;

  return false;
}

bool check_edge_bit_case_2(unsigned int forward, unsigned int backward, int player) {
  unsigned int cond_2a_fwd = pack_cells_3(player, EMPTY_SPACE, player);
  unsigned int cond_2a_bkwd = pack_cells_2(EMPTY_SPACE, player);

  unsigned int cond2_fwd = pack_cells_4(player, EMPTY_SPACE, player, EMPTY_SPACE);
  unsigned int cond2_bkwd = pack_cells_1(EMPTY_SPACE);

  unsigned int forward_three_cells = (forward >> 2) & 0x3F;
  unsigned int backward_one_cell = (backward >> 2) & 0x03;

  bool cond_2a = (cond_2a_fwd == forward_three_cells) && (cond_2a_bkwd == backward);

  if (!(cond_2a) && (cond2_fwd == forward) && (cond2_bkwd == backward_one_cell)) return true;

  return false;
}

bool check_edge_bit_case_3(unsigned int forward, unsigned int backward, int player) {
  unsigned int cond3_fwd = pack_cells_4(EMPTY_SPACE, player, player, EMPTY_SPACE);
  unsigned int cond3_bkwd = pack_cells_1(EMPTY_SPACE);

  unsigned int backward_one_cell = (backward >> 2) & 0x03;

  if ((cond3_fwd == forward) && (cond3_bkwd == backward_one_cell)) return true;

  return false;
}

bool check_edge_bit(Board &board, int x, int y, int dx, int dy, int player, int opponent) {
  // make sure to understand 4, 2 cells are getting acqured and partial
  // functions inside will shift values.
  unsigned int forward = board.extractLineAsBits(x, y, dx, dy, 4);
  unsigned int backward = board.extractLineAsBits(x, y, -dx, -dy, 2);

  if (Board::getCellCount(forward, 4) != 4 || Board::getCellCount(backward, 2) != 2)
    return false;  // out-of-bounds

  if (check_edge_bit_case_1(forward, backward, player, opponent)) return true;

  if (check_edge_bit_case_2(forward, backward, player)) return true;

  if (check_edge_bit_case_3(forward, backward, player)) return true;

  return false;
}

bool check_middle_bit_case_1(unsigned int forward, unsigned int backward, int player,
                             int opponent) {
  unsigned int cond1a_fwd = pack_cells_3(player, EMPTY_SPACE, opponent);
  unsigned int cond1a_bkwd = pack_cells_3(player, EMPTY_SPACE, opponent);

  unsigned int cond1b_fwd = pack_cells_3(player, EMPTY_SPACE, player);
  unsigned int cond1b_bkwd = pack_cells_1(player);

  unsigned int cond1c_fwd = pack_cells_2(player, EMPTY_SPACE);
  unsigned int cond1c_bkwd = pack_cells_2(player, EMPTY_SPACE);

  unsigned int forward_three_cells = forward & 0x3F;
  unsigned int backward_three_cells = backward & 0x3F;

  unsigned int forward_two_cells = (forward >> 2) & 0x0F;
  unsigned int backward_two_cells = (backward >> 2) & 0x0F;

  unsigned int backward_one_cell = (backward >> 4) & 0x03;

  unsigned int cond1a =
      (cond1a_fwd == forward_three_cells) && (cond1a_bkwd == backward_three_cells);
  unsigned int cond1b = (cond1b_fwd == forward_three_cells) && (cond1b_bkwd == backward_one_cell);
  unsigned int cond1c = (cond1c_fwd == forward_two_cells) && (cond1c_bkwd == backward_two_cells);

  if (!(cond1a || cond1b) && cond1c) return true;

  return false;
}

bool check_middle_bit_case_2(unsigned int forward, unsigned int backward, int player) {
  unsigned int cond2_fwd = pack_cells_3(EMPTY_SPACE, player, EMPTY_SPACE);
  unsigned int cond2_bkwd = pack_cells_2(player, EMPTY_SPACE);

  unsigned int backward_two_cells = (backward >> 2) & 0x0F;
  unsigned int forward_three_cells = forward & 0x3F;

  if ((cond2_fwd == forward_three_cells) && (cond2_bkwd == backward_two_cells)) return true;

  return false;
}

bool check_middle_bit_1(Board &board, int x, int y, int dx, int dy, int player, int opponent) {
  // make sure to understand only '3' cells are getting acqured and partial
  // functions inside will shift values.
  unsigned int forward = board.extractLineAsBits(x, y, dx, dy, 3);
  unsigned int backward = board.extractLineAsBits(x, y, -dx, -dy, 3);

  if (Board::getCellCount(forward, 3) != 3 || Board::getCellCount(backward, 3) != 3)
    return false;  // out-of-bounds

  if (check_middle_bit_case_1(forward, backward, player, opponent)) return true;

  return false;
}

bool check_middle_bit_2(Board &board, int x, int y, int dx, int dy, int player) {
  // make sure to understand only '3' cells are getting acqured and partial
  // functions inside will shift values.
  unsigned int forward = board.extractLineAsBits(x, y, dx, dy, 3);
  unsigned int backward = board.extractLineAsBits(x, y, -dx, -dy, 3);

  if (Board::getCellCount(forward, 3) != 3 || Board::getCellCount(backward, 3) != 3)
    return false;  // out-of-bounds

  if (check_middle_bit_case_2(forward, backward, player)) return true;

  return false;
}

// returns true if placing a stone at (x,y) for 'player' creates a double-three
// (bitwise).
bool Rules::detectDoublethreeBit(Board &board, int x, int y, int player) {
  int opponent = (player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  int count = 0;
  for (int i = 0; i < 8; ++i) {
    int dx = DIRECTIONS[i][0], dy = DIRECTIONS[i][1];
    if (!is_within_bounds(x, y, dx, dy)) continue;
    // if (board.getValueBit(x + dx, y + dy) == opponent) continue;
    if (check_edge_bit(board, x, y, dx, dy, player, opponent)) {
      ++count;
      continue;
    }
    if (i < 4 && check_middle_bit_1(board, x, y, dx, dy, player, opponent)) {
      ++count;
      continue;
    }
    if (check_middle_bit_2(board, x, y, dx, dy, player)) {
      ++count;
      continue;
    }
  }
  return (count >= 2);
}

bool Rules::isWinningMove(Board *board, int player, int x, int y) {
  if (MINIMAX_TERMINATION <= Evaluation::evaluatePositionHard(board, player, x, y)) return true;
  return false;
}
