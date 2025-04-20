#include "Board.hpp"

Board::Board()
    : goal(5),         // Default winning condition (e.g., 5 in a row)
      last_player(0),  // Default starting player for last move
      next_player(0),  // Default next player
      last_player_score(0),
      next_player_score(0),
      currentHash(0) {
  // Reset the bitboards to start with an empty board.
  resetBitboard();
  for (int row = 0; row < BOARD_SIZE; ++row) {
    for (int col = 0; col < BOARD_SIZE; ++col) {
      int index = row * BOARD_SIZE + col;
      currentHash ^= zobristTable[index][EMPTY_SPACE];
    }
  }
  currentHash ^= zobristTurn[getNextPlayer()];
}

Board::Board(const Board &other)
    : goal(other.goal),
      last_player(other.last_player),
      next_player(other.next_player),
      last_player_score(other.last_player_score),
      next_player_score(other.next_player_score),
      captured_stones(other.captured_stones),
      currentHash(other.currentHash) {
  for (int i = 0; i < BOARD_SIZE; ++i) {
    last_player_board[i] = other.last_player_board[i];
    next_player_board[i] = other.next_player_board[i];
  }
}

Board::Board(const std::vector<std::vector<char> > &board_data, int goal, int last_player_int,
             int next_player_int, int last_score, int next_score)
    : goal(goal),
      last_player(last_player_int),
      next_player(next_player_int),
      last_player_score(last_score),
      next_player_score(next_score) {
  this->resetBitboard();
  this->initBitboardFromData(board_data);

  // Recompute currentHash based on the new board state.
  currentHash = 0;
  for (int row = 0; row < BOARD_SIZE; ++row) {
    for (int col = 0; col < BOARD_SIZE; ++col) {
      int index = row * BOARD_SIZE + col;
      currentHash ^= zobristTable[index][getValueBit(col, row)];
    }
  }
  currentHash ^= zobristTurn[getNextPlayer()];
}

bool Board::isValidCoordinate(int col, int row) {
  return (col >= 0 && col < BOARD_SIZE && row >= 0 && row < BOARD_SIZE);
}

void Board::initBitboardFromData(const std::vector<std::vector<char> > &board_data) {
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

void Board::resetBitboard() {
  // Using memset (make sure ARRAY_SIZE * sizeof(uint64_t) is used)
  memset(this->last_player_board, 0, BOARD_SIZE * sizeof(uint64_t));
  memset(this->next_player_board, 0, BOARD_SIZE * sizeof(uint64_t));
}

uint64_t *Board::getBitboardByPlayer(int player) {
  if (player == PLAYER_1)
    return this->last_player_board;
  else if (player == PLAYER_2)
    return this->next_player_board;
  throw std::invalid_argument("Invalid player value");
}

// Set the cell (col, row) to a given player.
void Board::setValueBit(int col, int row, int stone) {
  if (!isValidCoordinate(col, row)) return;

  int index = row * BOARD_SIZE + col;
  int oldState = getValueBit(col, row);  // 0 = empty, 1 = P1, 2 = P2

  // 1) remove old piece if this is capture
  // 2) add new piece
  if (oldState != EMPTY_SPACE) currentHash ^= zobristTable[index][oldState];
  currentHash ^= zobristTable[index][stone];

  // stone: 0=empty (removal), 1=PLAYER_1, 2=PLAYER_2
  // update the bitboards (C++98)
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

  // 3) remove old turn, 4) flip, 5) add new turn
}

// Get the value at (col, row).
int Board::getValueBit(int col, int row) const {
  if (!isValidCoordinate(col, row)) return OUT_OF_BOUNDS;
  uint64_t mask = 1ULL << col;
  if (this->last_player_board[row] & mask) return PLAYER_1;
  if (this->next_player_board[row] & mask) return PLAYER_2;
  return EMPTY_SPACE;
}

void Board::getOccupancy(uint64_t occupancy[BOARD_SIZE]) const {
  for (int i = 0; i < BOARD_SIZE; i++) {
    occupancy[i] = last_player_board[i] | next_player_board[i];
  }
}

unsigned int Board::getCellCount(unsigned int pattern, int windowLength) {
  unsigned int count = 0;
  for (int i = 0; i < windowLength && (((pattern >> (2 * (windowLength - 1 - i))) & 0x3) != 3); ++i)
    ++count;
  return count;
}

/**
 * make sure to understand the function stores then shift the function stored
 * will be shifted to LEFT
 */
unsigned int Board::extractLineAsBits(int x, int y, int dx, int dy, int length) {
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

int Board::getNextPlayer() { return this->next_player; }

int Board::getLastPlayer() { return this->last_player; }

int Board::getNextPlayerScore() { return this->next_player_score; }

int Board::getLastPlayerScore() { return this->last_player_score; }

int Board::getGoal() { return this->goal; }

uint64_t Board::getHash() { return this->currentHash; }

const std::vector<CapturedStone> &Board::getCapturedStones() const { return this->captured_stones; }

// TODO: needs to change the hash value
void Board::switchTurn() {
  // 1) remember who was to move
  int oldTurn = next_player;

  // 2) swap players
  int tmp = next_player;
  next_player = last_player;
  last_player = tmp;

  // 3) swap scores
  tmp = next_player_score;
  next_player_score = last_player_score;
  last_player_score = tmp;

  // 4) update zobrist hash: remove old-turn, add new-turn
  currentHash ^= zobristTurn[oldTurn];
  currentHash ^= zobristTurn[next_player];
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

std::pair<int, int> Board::getCurrentScore() {
  std::pair<int, int> ret;

  ret.first = this->last_player_score;
  ret.second = this->next_player_score;
  return ret;
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

void Board::storeCapturedStone(int x, int y, int player) {
  CapturedStone cs;
  cs.x = x;
  cs.y = y;
  cs.player = player;
  // Use push_back since captured_stones is a std::vector
  captured_stones.push_back(cs);
}

void Board::updateLastPlayerScore(int newAddedScore) {
  // Remove the old last_player_score's contribution from the hash.
  currentHash ^= static_cast<uint64_t>(this->last_player_score) * LAST_SCORE_MULTIPLIER;
  // Update the score variable.
  this->last_player_score += newAddedScore;
  // Add the new score's contribution.
  currentHash ^= static_cast<uint64_t>(this->last_player_score) * LAST_SCORE_MULTIPLIER;
}

void Board::updateNextPlayerScore(int newAddedScore) {
  // Remove the old next_player_score's contribution.
  currentHash ^= static_cast<uint64_t>(this->next_player_score) * NEXT_SCORE_MULTIPLIER;
  // Update the score.
  this->next_player_score += newAddedScore;
  // Add the new score's contribution.
  currentHash ^= static_cast<uint64_t>(this->next_player_score) * NEXT_SCORE_MULTIPLIER;
}

void Board::applyCapture(bool clearCapture) {
  int newLastPlayerScore = 0;
  int newNextPlayerScore = 0;
  for (size_t i = 0; i < captured_stones.size(); ++i) {
    // std::cout << "[" << captured_stones[i].x << "][" << captured_stones[i].y << "]["
    //           << captured_stones[i].player << "]" << std::endl;
    if (captured_stones[i].player == last_player) {
      // std::cout << "checking" << std::endl;
      newNextPlayerScore++;
    } else if (captured_stones[i].player == next_player) {
      newLastPlayerScore++;
      // std::cout << "checking 2" << std::endl;
    }

    setValueBit(captured_stones[i].x, captured_stones[i].y, EMPTY_SPACE);
  }

  newLastPlayerScore /= 2;
  newNextPlayerScore /= 2;

  updateLastPlayerScore(newLastPlayerScore);
  updateNextPlayerScore(newNextPlayerScore);

  if (clearCapture) captured_stones.clear();
}
