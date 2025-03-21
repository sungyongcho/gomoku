#include "minimax.hpp"

#include <cstdlib>
#include <ctime>
#include <limits>
#include <sstream>

namespace Minimax {
int combinedPatternScoreTable[LOOKUP_TABLE_SIZE] = {0};

// (TODO) needs to be improved, this only checks the basic continous pattern
int evaluateCombinedPattern(int combinedPattern, int player) {
  int opponent = OPPONENT(player);
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

  if (cells[center + 1] == opponent && cells[center + 2] == opponent && cells[center + 3] == player)
    return CAPTURE_SCORE;

  if (cells[center - 1] == opponent && cells[center - 2] == opponent && cells[center - 3] == player)
    return CAPTURE_SCORE;
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
  // int opponent = (player == PLAYER_1 ? PLAYER_2 : PLAYER_1);
  // // Forward capture check: if cells at indices center+1, center+2, center+3
  // // equal: opponent, opponent, player.
  // if (center + 3 < COMBINED_WINDOW_SIZE) {
  //   if (cells[center + 1] == opponent && cells[center + 2] == opponent &&
  //       cells[center + 3] == player) {
  //     score += CAPTURE_SCORE;
  //   }
  // }
  // // Backward capture check: if cells at indices center-3, center-2, center-1
  // // equal: player, opponent, opponent.
  // if (center - 3 >= 0) {
  //   if (cells[center - 1] == opponent && cells[center - 2] == opponent &&
  //       cells[center - 3] == player) {
  //     score += CAPTURE_SCORE;
  //   }
  // }
  return score;
}

void initCombinedPatternScoreTable() {
  for (int pattern = 0; pattern < LOOKUP_TABLE_SIZE; pattern++) {
    // Here we assume evaluation for PLAYER_1.
    // (For two-player support, either build two tables or adjust at runtime.)
    combinedPatternScoreTable[pattern] = evaluateCombinedPattern(pattern, PLAYER_1);
    combinedPatternScoreTable[pattern] = evaluateCombinedPattern(pattern, PLAYER_2);
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

struct MoveComparatorMax {
  const Board *board;
  int player;
  MoveComparatorMax(const Board *b, int p) : board(b), player(p) {}

  bool operator()(const std::pair<int, int> &m1, const std::pair<int, int> &m2) const {
    // For move ordering, we can use a lightweight evaluation.
    // Here, we call evaluatePosition on clones of the board.
    Board *child1 = Board::cloneBoard(board);
    child1->setValueBit(m1.first, m1.second, player);
    int score1 = evaluatePosition(child1, player, m1.first, m1.second);
    delete child1;

    Board *child2 = Board::cloneBoard(board);
    child2->setValueBit(m2.first, m2.second, player);
    int score2 = evaluatePosition(child2, player, m2.first, m2.second);
    delete child2;

    return score1 > score2;  // For maximizer: higher score first.
  }
};

struct MoveComparatorMin {
  const Board *board;
  int player;
  MoveComparatorMin(const Board *b, int p) : board(b), player(p) {}

  bool operator()(const std::pair<int, int> &m1, const std::pair<int, int> &m2) const {
    Board *child1 = Board::cloneBoard(board);
    child1->setValueBit(m1.first, m1.second, player);
    int score1 = evaluatePosition(child1, player, m1.first, m1.second);
    delete child1;

    Board *child2 = Board::cloneBoard(board);
    child2->setValueBit(m2.first, m2.second, player);
    int score2 = evaluatePosition(child2, player, m2.first, m2.second);
    delete child2;

    return score1 < score2;  // For minimizer: lower score first.
  }
};

// Filter candidate moves to remove moves that cause a double-three.
// Assume detectDoublethree is implemented to return true if the move creates a double-three.
std::vector<std::pair<int, int> > filterDoubleThreeMoves(
    Board *board, const std::vector<std::pair<int, int> > &moves, int player) {
  std::vector<std::pair<int, int> > filtered;
  for (size_t i = 0; i < moves.size(); i++) {
    Board *temp = Board::cloneBoard(board);
    temp->setValueBit(moves[i].first, moves[i].second, player);
    // If this move does not create a double-three, keep it.
    if (!Rules::detectDoublethreeBit(*temp, moves[i].first, moves[i].second, player)) {
      filtered.push_back(moves[i]);
    }
    delete temp;
  }
  return filtered;
}

int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY) {
  // if (Rules::isWinningMove(board, PLAYER_1, lastX, lastY))
  //   return std::numeric_limits<int>::max();  // or GOMOKU
  // if (Rules::isWinningMove(board, PLAYER_2, lastX, lastY))
  //   return -std::numeric_limits<int>::max();  // or -GOMOKU

  // Base case: depth 0, evaluate the board based on the last move.
  if (depth == 0) {
    return evaluatePosition(board, currentPlayer, lastX, lastY);
  }

  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    return evaluatePosition(board, currentPlayer, lastX, lastY);
  }
  moves = filterDoubleThreeMoves(board, moves, currentPlayer);

  // --- Apply selective search: sort moves using quick evaluation.
  if (currentPlayer == PLAYER_1) {  // Maximizer
    std::sort(moves.begin(), moves.end(), MoveComparatorMax(board, currentPlayer));
    const size_t TOP_N = 10;
    if (moves.size() > TOP_N) moves.resize(TOP_N);
  } else {
    std::sort(moves.begin(), moves.end(), MoveComparatorMin(board, currentPlayer));
    const size_t TOP_N = 10;
    if (moves.size() > TOP_N) moves.resize(TOP_N);
  }

  if (currentPlayer == PLAYER_1) {  // Maximizing player.
    int maxEval = -std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); i++) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      if (Rules::detectCaptureStones(*child, moves[i].first, moves[i].second, currentPlayer))
        child->applyCapture();
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
      if (Rules::detectCaptureStones(*child, moves[i].first, moves[i].second, currentPlayer))
        child->applyCapture();
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
  moves = filterDoubleThreeMoves(board, moves, player);

  // --- Apply selective search: sort moves using quick evaluation.
  if (player == PLAYER_1) {  // Maximizer
    std::sort(moves.begin(), moves.end(), MoveComparatorMax(board, player));
    const size_t TOP_N = 10;
    if (moves.size() > TOP_N) moves.resize(TOP_N);
  } else {
    std::sort(moves.begin(), moves.end(), MoveComparatorMin(board, player));
    const size_t TOP_N = 10;
    if (moves.size() > TOP_N) moves.resize(TOP_N);
  }

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

void simulateAIBattle(Board *pBoard, int searchDepth, int numTurns) {
  int currentPlayer = pBoard->getNextPlayer();  // or choose starting player

  for (int turn = 0; turn < numTurns; ++turn) {
    std::clock_t start = std::clock();  // Start time
    // Get the best move for the current player.
    std::pair<int, int> bestMove = Minimax::getBestMove(pBoard, currentPlayer, searchDepth);
    if (bestMove.first == -1) {
      std::cout << "No valid move found. Ending simulation." << std::endl;
      break;
    }
    // Apply the move.
    pBoard->setValueBit(bestMove.first, bestMove.second, currentPlayer);
    std::clock_t end = std::clock();  // End time

    std::cout << "Turn " << turn + 1 << ": Player " << currentPlayer << " moves at ("
              << bestMove.first << ", " << bestMove.second << ")" << std::endl;
    pBoard->printBitboard();
    std::cout << "---------------------------" << std::endl;
    double elapsed_seconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    double elapsed_ms = elapsed_seconds * 1000.0;
    double elapsed_ns = elapsed_seconds * 1e9;

    std::cout << "Execution time: " << elapsed_seconds << " s, " << elapsed_ms << " ms, "
              << elapsed_ns << " ns" << std::endl;
    std::cout << "---------------------------" << std::endl;

    // Switch players.
    currentPlayer = (currentPlayer == PLAYER_1 ? PLAYER_2 : PLAYER_1);
  }
}

}  // namespace Minimax
