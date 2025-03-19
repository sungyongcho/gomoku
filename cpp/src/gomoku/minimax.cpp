#include "minimax.hpp"

#include <cstdlib>
#include <limits>
#include <sstream>

namespace Minimax {
int combinedPatternScoreTable[LOOKUP_TABLE_SIZE] = {0};

// (TODO) needs to be improved, this only checks the basic continous pattern
int evaluateCombinedPattern(int combinedPattern, int player) {
  int cells[COMBINED_WINDOW_SIZE];
  // Decode each cell (2 bits per cell).
  for (int i = 0; i < COMBINED_WINDOW_SIZE; i++) {
    int shift = 2 * (COMBINED_WINDOW_SIZE - 1 - i);
    cells[i] = (combinedPattern >> shift) & 0x3;
  }
  // The center index is SIDE_WINDOW_SIZE.
  int center = SIDE_WINDOW_SIZE;
  // Ensure the center cell is set to player's stone.
  cells[center] = player;

  // Count contiguous stones including the center.
  int leftCount = 0;
  for (int i = center - 1; i >= 0; i--) {
    if (cells[i] == player)
      leftCount++;
    else
      break;
  }
  int rightCount = 0;
  for (int i = center + 1; i < COMBINED_WINDOW_SIZE; i++) {
    if (cells[i] == player)
      rightCount++;
    else
      break;
  }
  int totalRun = leftCount + 1 + rightCount;

  // Check open ends: if the cell immediately outside the contiguous run is
  // EMPTY.
  bool openLeft = (center - leftCount - 1 >= 0 && cells[center - leftCount - 1] == EMPTY_SPACE);
  bool openRight = (center + rightCount + 1 < COMBINED_WINDOW_SIZE &&
                    cells[center + rightCount + 1] == EMPTY_SPACE);

  int score = 0;
  if (totalRun >= 5)
    score = GOMOKU;
  else if (totalRun == 4)
    score = (openLeft && openRight) ? OPEN_LINE_4 : BLOCKED_LINE_4;
  else if (totalRun == 3)
    score = (openLeft && openRight) ? OPEN_LINE_3 : BLOCKED_LINE_3;
  else if (totalRun == 2)
    score = (openLeft && openRight) ? OPEN_LINE_2 : BLOCKED_LINE_2;
  else
    score = (openLeft && openRight) ? OPEN_SINGLE_STONE : 0;

  // Check capture opportunities.
  int opponent = (player == PLAYER_1 ? PLAYER_2 : PLAYER_1);
  // Forward capture check: if cells at indices center+1, center+2, center+3
  // equal: opponent, opponent, player.
  if (center + 3 < COMBINED_WINDOW_SIZE) {
    if (cells[center + 1] == opponent && cells[center + 2] == opponent &&
        cells[center + 3] == player) {
      score += CAPTURE_SCORE;
    }
  }
  // Backward capture check: if cells at indices center-3, center-2, center-1
  // equal: player, opponent, opponent.
  if (center - 3 >= 0) {
    if (cells[center - 1] == opponent && cells[center - 2] == opponent &&
        cells[center - 3] == player) {
      score += CAPTURE_SCORE;
    }
  }
  return score;
}

void initCombinedPatternScoreTable() {
  for (int pattern = 0; pattern < LOOKUP_TABLE_SIZE; pattern++) {
    // Here we assume evaluation for PLAYER_1.
    // (For two-player support, either build two tables or adjust at runtime.)
    combinedPatternScoreTable[pattern] = evaluateCombinedPattern(pattern, PLAYER_1);
  }
}

inline unsigned int reversePattern(unsigned int pattern, int windowSize) {
  unsigned int reversed = 0;
  for (int i = 0; i < windowSize; i++) {
    reversed = (reversed << 2) | (pattern & 0x3);
    pattern >>= 2;
  }
  return reversed;
}

int evaluateCombinedAxis(Board *board, int player, int x, int y, int dx, int dy) {
  // Extract the forward window.
  unsigned int forward = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  // Extract the backward window.
  unsigned int backward = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  // Reverse the backward window so that the cell immediately adjacent to (x,y)
  // is at the rightmost position.
  unsigned int revBackward = reversePattern(backward, SIDE_WINDOW_SIZE);
  // Combine: [reversed backward window] + [center cell (player)] + [forward
  // window]
  unsigned int combined = (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) |
                          ((unsigned int)player << (2 * SIDE_WINDOW_SIZE)) | forward;
  int score = combinedPatternScoreTable[combined];
  return score;
}

int evaluatePosition(Board *&board, int player, int x, int y) {
  int totalScore = 0;

  if (board->getValueBit(x, y) == EMPTY_SPACE) return 0;

  for (int i = 0; i < 4; ++i)
    totalScore += evaluateCombinedAxis(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);

  return totalScore;
}

static const uint64_t rowMask = ((uint64_t)1 << BOARD_SIZE) - 1;

// Horizontal shifts for a row.
inline uint64_t shiftRowLeft(uint64_t row) { return (row << 1) & rowMask; }
inline uint64_t shiftRowRight(uint64_t row) { return row >> 1; }

// Compute neighbor mask for each row based on occupancy.
void computeNeighborMask(const uint64_t occupancy[BOARD_SIZE], uint64_t neighbor[BOARD_SIZE]) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    uint64_t row = occupancy[i];
    uint64_t horz = shiftRowLeft(row) | shiftRowRight(row) | row;
    uint64_t vert = 0;
    if (i > 0) vert |= occupancy[i - 1];
    if (i < BOARD_SIZE - 1) vert |= occupancy[i + 1];
    uint64_t diag = 0;
    if (i > 0) {
      diag |= shiftRowLeft(occupancy[i - 1]);
      diag |= shiftRowRight(occupancy[i - 1]);
    }
    if (i < BOARD_SIZE - 1) {
      diag |= shiftRowLeft(occupancy[i + 1]);
      diag |= shiftRowRight(occupancy[i + 1]);
    }
    neighbor[i] = horz | vert | diag;
  }
}

// Generate candidate moves using row-based neighbor mask.
std::vector<std::pair<int, int> > generateCandidateMoves(Board *&board) {
  std::vector<std::pair<int, int> > moves;
  uint64_t occupancy[BOARD_SIZE];
  uint64_t neighbor[BOARD_SIZE];

  board->getOccupancy(occupancy);
  computeNeighborMask(occupancy, neighbor);

  for (int row = 0; row < BOARD_SIZE; row++) {
    uint64_t candidates = neighbor[row] & (~occupancy[row]) & rowMask;
    for (int col = 0; col < BOARD_SIZE; col++) {
      if (candidates & (1ULL << col)) moves.push_back(std::make_pair(col, row));
    }
  }
  return moves;
}

void printBoardWithCandidates(Board *&board, const std::vector<std::pair<int, int> > &candidates) {
  // Create a 2D display grid.
  std::vector<std::vector<char> > display(BOARD_SIZE, std::vector<char>(BOARD_SIZE, '.'));
  for (int y = 0; y < BOARD_SIZE; y++) {
    for (int x = 0; x < BOARD_SIZE; x++) {
      int val = board->getValueBit(x, y);
      if (val == PLAYER_1)
        display[y][x] = '1';
      else if (val == PLAYER_2)
        display[y][x] = '2';
    }
  }
  // Mark candidate moves with 'C' (if the cell is empty).
  for (size_t i = 0; i < candidates.size(); i++) {
    int x = candidates[i].first, y = candidates[i].second;
    if (board->getValueBit(x, y) == EMPTY_SPACE) display[y][x] = 'C';
  }
  // Print the board.
  for (int y = 0; y < BOARD_SIZE; y++) {
    for (int x = 0; x < BOARD_SIZE; x++) {
      std::cout << display[y][x] << " ";
    }
    std::cout << "\n";
  }

  std::cout << std::flush;
}

int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY) {
  // Base case: depth 0, evaluate the board based on the last move.
  if (depth == 0) {
    return evaluatePosition(board, currentPlayer, lastX, lastY);
  }

  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    return evaluatePosition(board, currentPlayer, lastX, lastY);
  }

  if (currentPlayer == PLAYER_1) {  // Maximizing player.
    int maxEval = -std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); i++) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      int eval = minimax(child, depth - 1, alpha, beta, PLAYER_2, moves[i].first, moves[i].second);
      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) break;  // Beta cutoff.
    }
    return maxEval;
  } else {  // Minimizing player.
    int minEval = std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); i++) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      int eval = minimax(child, depth - 1, alpha, beta, PLAYER_1, moves[i].first, moves[i].second);
      minEval = std::min(minEval, eval);
      beta = std::min(beta, eval);
      if (beta <= alpha) break;  // Alpha cutoff.
    }
    return minEval;
  }
}

std::pair<int, int> getBestMove(Board *board, int player, int depth) {
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  std::pair<int, int> bestMove = std::make_pair(-1, -1);
  if (moves.empty()) return bestMove;

  int bestScore;
  if (player == PLAYER_1) {  // Maximizer.
    bestScore = -std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); i++) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, player);
      int score =
          minimax(child, depth - 1, -std::numeric_limits<int>::max(),
                  std::numeric_limits<int>::max(), PLAYER_2, moves[i].first, moves[i].second);
      if (score > bestScore) {
        bestScore = score;
        bestMove = moves[i];
      }
    }
  } else {  // Minimizer.
    bestScore = std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); i++) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, player);
      int score =
          minimax(child, depth - 1, -std::numeric_limits<int>::max(),
                  std::numeric_limits<int>::max(), PLAYER_1, moves[i].first, moves[i].second);
      if (score < bestScore) {
        bestScore = score;
        bestMove = moves[i];
      }
    }
  }
  std::cout << "score: " << bestScore << std::endl;
  std::cout << "bestMove: " << bestMove.first << ", " << bestMove.second << std::endl;
  return bestMove;
}

}  // namespace Minimax
