#include "Board.hpp"

#include "Rules.hpp"

Board::Board()
    : goal(5),
      last_player(PLAYER_1),
      next_player(PLAYER_2),
      last_player_score(0),
      next_player_score(0),
      currentHash(0) {
  assert(Zobrist::initialized &&
         "Zobrist keys must be initialized before creating a Board object!");

  this->reset_bitboard();
  currentHash ^= Zobrist::capture_keys[PLAYER_1][0];  // Black starts with 0 points
  currentHash ^= Zobrist::capture_keys[PLAYER_2][0];  // White starts with 0 points
  if (this->next_player == PLAYER_2) {                // Check if P2 starts by default
    currentHash ^= Zobrist::turn_key;
  }
}

Board::Board(const Board &other)
    : goal(other.goal),
      last_player(other.last_player),
      next_player(other.next_player),
      last_player_score(other.last_player_score),
      next_player_score(other.next_player_score),
      currentHash(other.currentHash) {
  for (int i = 0; i < BOARD_SIZE; ++i) {
    last_player_board[i] = other.last_player_board[i];
    next_player_board[i] = other.next_player_board[i];
  }
}

Board::Board(const std::vector<std::vector<char> > &board_data, int goal, int last_player_int,
             int next_player_int, int last_score, int next_score, bool enableCapture,
             bool enableDoubleThreeRestriction)
    : goal(goal),
      last_player(last_player_int),
      next_player(next_player_int),
      last_player_score(last_score),
      next_player_score(next_score),
      enable_capture(enableCapture),
      enable_double_three_restriction(enableDoubleThreeRestriction),
      currentHash(0) {
  assert(Zobrist::initialized &&
         "Zobrist keys must be initialized before creating a Board object!");

  this->reset_bitboard();
  this->init_bitboard_from_data(board_data);

  currentHash ^= Zobrist::capture_keys[this->last_player][this->last_player_score];
  currentHash ^= Zobrist::capture_keys[this->next_player][this->next_player_score];

  if (this->next_player == PLAYER_2) {
    currentHash ^= Zobrist::turn_key;
  }
}

Board::Board(int goal, int last_player_int, int next_player_int, int last_score, int next_score,
             bool enableCapture, bool enableDoubleThreeRestriction)
    : goal(goal),
      last_player(last_player_int),
      next_player(next_player_int),
      last_player_score(last_score),
      next_player_score(next_score),
      enable_capture(enableCapture),
      enable_double_three_restriction(enableDoubleThreeRestriction),
      currentHash(0) {
  assert(Zobrist::initialized &&
         "Zobrist keys must be initialized before creating a Board object!");

  this->reset_bitboard();

  currentHash ^= Zobrist::capture_keys[this->last_player][this->last_player_score];
  currentHash ^= Zobrist::capture_keys[this->next_player][this->next_player_score];

  if (this->next_player == PLAYER_2) {
    currentHash ^= Zobrist::turn_key;
  }
}

/**
 * Private Methods
 */
void Board::reset_bitboard() {
  memset(this->last_player_board, 0, BOARD_SIZE * sizeof(uint64_t));
  memset(this->next_player_board, 0, BOARD_SIZE * sizeof(uint64_t));
}

void Board::init_bitboard_from_data(const std::vector<std::vector<char> > &board_data) {
  for (size_t r = 0; r < board_data.size(); ++r) {
    for (size_t c = 0; c < board_data[r].size(); ++c) {
      if (!isValidCoordinate(c, r)) continue;
      if (board_data[r][c] == PLAYER_X) {
        setValueBit(c, r, PLAYER_1);
      } else if (board_data[r][c] == PLAYER_O) {
        setValueBit(c, r, PLAYER_2);
      }
    }
  }
}

/**
 * Accessors
 */
int Board::getValueBit(int col, int row) const {
  if (!isValidCoordinate(col, row)) return OUT_OF_BOUNDS;
  uint64_t mask = 1ULL << col;
  if (this->last_player_board[row] & mask) return PLAYER_1;
  if (this->next_player_board[row] & mask) return PLAYER_2;
  return EMPTY_SPACE;
}

uint64_t *Board::getBitboardByPlayer(int player) {
  if (player == PLAYER_1)
    return this->last_player_board;
  else if (player == PLAYER_2)
    return this->next_player_board;
  throw std::invalid_argument("Invalid player value");
}

void Board::getOccupancy(uint64_t occupancy[BOARD_SIZE]) const {
  for (int i = 0; i < BOARD_SIZE; i++) {
    occupancy[i] = last_player_board[i] | next_player_board[i];
  }
}

uint64_t Board::getHash() const { return this->currentHash; }

int Board::getNextPlayer() const { return this->next_player; }

int Board::getLastPlayer() const { return this->last_player; }

int Board::getNextPlayerScore() const { return this->next_player_score; }

int Board::getLastPlayerScore() const { return this->last_player_score; }
std::pair<int, int> Board::getCurrentScore() const {
  std::pair<int, int> ret;

  ret.first = this->last_player_score;
  ret.second = this->next_player_score;
  return ret;
}

int Board::getGoal() const { return this->goal; }

bool Board::getEnableCapture() const { return this->enable_capture; }
bool Board::getEnableDoubleThreeRestriction() const {
  return this->enable_double_three_restriction;
}

/**
 * Executions
 */
void Board::setValueBit(int col, int row, int stone) {
  if (!isValidCoordinate(col, row)) return;

  int old_player_at_cell = this->getValueBit(col, row);  // Will be 0, 1, or 2

  // Only update the hash if the state is actually changing.
  if (old_player_at_cell != stone) {
    if (old_player_at_cell == PLAYER_1) {
      this->currentHash ^= Zobrist::piece_keys[col][row][PLAYER_1];
    } else if (old_player_at_cell == PLAYER_2) {
      this->currentHash ^= Zobrist::piece_keys[col][row][PLAYER_2];
    }
    if (stone == PLAYER_1) {
      this->currentHash ^= Zobrist::piece_keys[col][row][PLAYER_1];
    } else if (stone == PLAYER_2) {
      this->currentHash ^= Zobrist::piece_keys[col][row][PLAYER_2];
    }
  }

  // // stone: 0=empty (removal), 1=PLAYER_1, 2=PLAYER_2
  // // update the bitboards (C++98)
  uint64_t mask = 1ULL << col;
  if (stone == 0) {
    // removal: clear that cell on both bitboards
    last_player_board[row] &= ~mask;
    next_player_board[row] &= ~mask;
  } else if (stone == PLAYER_1) {
    // placement of P1: just set the bit, don't clear first
    last_player_board[row] |= mask;
  } else if (stone == PLAYER_2) {
    // placement of P2
    next_player_board[row] |= mask;
  }
}

void Board::storeCapturedStone(int x, int y, int player) {
  CapturedStone cs;
  cs.x = x;
  cs.y = y;
  cs.player = player;
  // Use push_back since captured_stones is a std::vector
  captured_stones.push_back(cs);
}

void Board::applyCapture(bool clearCapture) {
  int opponent_stones_removed_count = 0;

  for (std::vector<CapturedStone>::size_type i = 0; i < captured_stones.size(); ++i) {
    const CapturedStone &cap = captured_stones[i];  // Use const reference

    // Check if the captured stone belonged to the opponent
    if (cap.player == this->next_player) {
      opponent_stones_removed_count++;
    }

    this->setValueBit(cap.x, cap.y, EMPTY_SPACE);
  }

  int pairs_captured = opponent_stones_removed_count / 2;
  if (pairs_captured > 0) {
    int capturing_player = this->last_player;  // Player who just moved gets points
    int old_score = this->last_player_score;
    int new_total_score = old_score + pairs_captured;

    if (new_total_score > this->goal) {
      new_total_score = this->goal;
    }

    if (new_total_score != old_score) {
      // Update Zobrist hash for the score change:
      // XOR out the key for the old score, XOR in the key for the new score.
      this->currentHash ^= Zobrist::capture_keys[capturing_player][old_score];
      this->currentHash ^= Zobrist::capture_keys[capturing_player][new_total_score];

      // Update the player's score
      this->last_player_score = new_total_score;
    }
  }
  if (clearCapture) captured_stones.clear();
}

const std::vector<CapturedStone> &Board::getCapturedStones() const { return this->captured_stones; }

void Board::switchTurn() {
  int tmp = this->next_player;
  this->next_player = this->last_player;
  this->last_player = tmp;

  int tmp_score = this->next_player_score;
  this->next_player_score = this->last_player_score;
  this->last_player_score = tmp_score;

  this->currentHash ^= Zobrist::turn_key;
}

void Board::flushCaptures() {
  if (this->getCapturedStones().size() > 0) {
    this->applyCapture(true);
  }
}

/**
 * Utility
 */
bool Board::isValidCoordinate(int col, int row) {
  return (col >= 0 && col < BOARD_SIZE && row >= 0 && row < BOARD_SIZE);
}

std::string Board::convertIndexToCoordinates(int col, int row) {
  if (col < 0 || col >= 19) {
    throw std::out_of_range("Column index must be between 0 and 18.");
  }
  if (row < 0 || row >= 19) {
    throw std::out_of_range("Row index must be between 0 and 18.");
  }

  char colChar = 'A' + col;  // Convert 0-18 to 'A'-'S'

  std::stringstream ss;
  ss << (row + 1);  // Convert 0-18 to 1-19 and convert to string

  return std::string(1, colChar) + ss.str();
}

/**
 * make sure to understand the function stores then shift the function stored
 * will be shifted to LEFT
 */
unsigned int Board::extractLineAsBits(int x, int y, int dx, int dy, int length) const {
  unsigned int pattern = 0;
  // Loop from 1 to 'length'
  for (int i = 1; i <= length; ++i) {
    // Update coordinates incrementally.
    x += dx;
    y += dy;
    // Check if within bounds.
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
      pattern = (pattern << 2) | (OUT_OF_BOUNDS & 0x3);
    } else {
      int cell = this->getValueBit(x, y);  // Returns 0, 1, or 2.
      // Pack the cell value into the pattern (using 2 bits per cell).
      pattern = (pattern << 2) | (cell & 0x3);
      // cell & 0x3 ensures that only the lower 2 bits of 'cell' are kept.
      // The mask 0x3 is binary 11 (i.e., 0b11), so any value in 'cell' will be
      // reduced to its two least-significant bits, effectively restricting the
      // result to one of four possible values (0-3). In our usage, we expect
      // cell values to be 0 (empty), 1 (PLAYER_1), or 2 (PLAYER_2).
      //
      // For example, if cell = 5 (binary 101):
      //      101 (binary for 5)
      //   &  011 (binary for 0x3)
      //   ---------
      //      001 (binary for 1)
      // Thus, 5 & 0x3 yields 1, ensuring that any extraneous higher bits are
      // ignored.
    }
  }
  return pattern;
}

unsigned int Board::getCellCount(unsigned int pattern, int windowLength) {
  unsigned int count = 0;
  for (int i = 0; i < windowLength && (((pattern >> (2 * (windowLength - 1 - i))) & 0x3) != 3); ++i)
    ++count;
  return count;
}

void Board::printBitboard() const {
  for (int r = 0; r < BOARD_SIZE; r++) {
    for (int c = 0; c < BOARD_SIZE; c++) {
      int v = getValueBit(c, r);
      if (v == PLAYER_1)
        std::cout << "1 ";
      else if (v == PLAYER_2)
        std::cout << "2 ";
      else if (v == OUT_OF_BOUNDS)
        std::cout << "X ";
      else
        std::cout << ". ";
    }
    std::cout << std::endl;
  }
}

/**
 * For Debug
 */
void Board::BitboardToJsonBoardboard(rapidjson::Value &json_board,
                                     rapidjson::Document::AllocatorType &allocator) const {
  json_board.SetArray();  // Ensure it's an array type

  for (int r = 0; r < BOARD_SIZE; ++r) {
    rapidjson::Value json_row(rapidjson::kArrayType);
    for (int c = 0; c < BOARD_SIZE; ++c) {
      rapidjson::Value cell;
      char temp_str[2];
      int value = getValueBit(c, r);
      if (value == PLAYER_1)
        temp_str[0] = PLAYER_X;  // 'X'
      else if (value == PLAYER_2)
        temp_str[0] = PLAYER_O;  // 'O'
      else
        temp_str[0] = '.';  // empty
      temp_str[1] = '\0';
      cell.SetString(temp_str, allocator);
      json_row.PushBack(cell, allocator);
    }
    json_board.PushBack(json_row, allocator);
  }
}

// Helper: Print a bit-packed line pattern (reversed if needed)
void print_line_pattern_impl(unsigned int pattern, int length, bool reversed) {
  std::cout << (reversed ? "pattern (reversed): " : "pattern: ") << "[";
  for (int i = 0; i < length; ++i) {
    int shift = reversed ? 2 * i : 2 * (length - i - 1);
    int cell = (pattern >> shift) & 0x3;
    char symbol = (cell == 0)   ? '.'
                  : (cell == 1) ? '1'
                  : (cell == 2) ? '2'
                  : (cell == 3) ? 'X'
                                : '?';
    std::cout << symbol << " ";
  }
  std::cout << "]" << std::endl;
}

void printLinePatternReverse(unsigned int pattern, int length) {
  print_line_pattern_impl(pattern, length, true);
}

void Board::printLinePattern(unsigned int pattern, int length) {
  print_line_pattern_impl(pattern, length, false);
}

UndoInfo Board::makeMove(int col, int row) {
  UndoInfo undo_data;

  int player = this->next_player;
  if (!isValidCoordinate(col, row) || this->getValueBit(col, row) != EMPTY_SPACE)
    throw std::runtime_error("Invalid move: Square occupied or out of bounds");

  undo_data.move = std::make_pair(col, row);

  if (player == this->next_player)
    undo_data.scoreBeforeCapture = this->next_player_score;
  else  // Should technically be the next_player, but handle if state is unusual
    undo_data.scoreBeforeCapture = this->last_player_score;  // Or fetch based on 'player' ID

  this->setValueBit(col, row, player);

  if (Rules::detectCaptureStones(*this, col, row, player)) {
    undo_data.capturedStonesInfo = this->captured_stones;
  }
  this->switchTurn();
  return undo_data;
}
void Board::undoMove(const UndoInfo &undo_data) {
  this->switchTurn();  // Switch back to the player who made the move

  int player_who_moved = this->next_player;  // Player is now the one who moved

  // Undo stone placement
  this->setValueBit(undo_data.move.first, undo_data.move.second, EMPTY_SPACE);
  // Update hash for stone removal (assuming setValueBit handles this)

  // Undo captures if any occurred
  if (!undo_data.capturedStonesInfo.empty()) {
    // int score_change = (undo_data.capturedStonesInfo.size() / 2);  // Calculate score delta
    int score_after_capture = this->next_player_score;  // Score before undoing capture points
    // int score_before_capture =
    //     score_after_capture - score_change;  // Calculate the score before capture

    // Update Zobrist hash for score change *before* changing the score variable
    this->currentHash ^= Zobrist::capture_keys[player_who_moved][score_after_capture];
    this->currentHash ^= Zobrist::capture_keys
        [player_who_moved][undo_data.scoreBeforeCapture];  // Assumes keys are for absolute scores

    // Restore the player's score member variable
    this->next_player_score = undo_data.scoreBeforeCapture;

    // Restore the captured stones on the board
    for (std::vector<CapturedStone>::const_iterator it = undo_data.capturedStonesInfo.begin();
         it != undo_data.capturedStonesInfo.end(); it++) {
      const CapturedStone &cap = *it;
      // Assuming setValueBit updates the hash for placing the stone back
      this->setValueBit(cap.x, cap.y, cap.player);
    }
  }
  // Note: Removed switchTurn() from the end, it should only be at the beginning.
}
