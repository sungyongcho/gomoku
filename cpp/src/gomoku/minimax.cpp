#include "minimax.hpp"

#include <cstdlib>
#include <ctime>
#include <limits>
#include <sstream>

namespace Minimax {

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
    int score1 = Evaluation::evaluatePosition(child1, player, m1.first, m1.second);
    delete child1;

    Board *child2 = Board::cloneBoard(board);
    child2->setValueBit(m2.first, m2.second, player);
    int score2 = Evaluation::evaluatePosition(child2, player, m2.first, m2.second);
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
    int score1 = Evaluation::evaluatePosition(child1, player, m1.first, m1.second);
    delete child1;

    Board *child2 = Board::cloneBoard(board);
    child2->setValueBit(m2.first, m2.second, player);
    int score2 = Evaluation::evaluatePosition(child2, player, m2.first, m2.second);
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

int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY,
            bool isMaximizing) {
  // Terminal condition: if we've reached the maximum search depth.
  if (depth == 0) {
    // Evaluate board based on the last move (played by opponent of currentPlayer)
    return Evaluation::evaluatePosition(board, OPPONENT(currentPlayer), lastX, lastY);
  }

  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty())
    return Evaluation::evaluatePosition(board, OPPONENT(currentPlayer), lastX, lastY);

  if (isMaximizing) {
    int maxEval = std::numeric_limits<int>::min();
    for (size_t i = 0; i < moves.size(); ++i) {
      // Create a child board state.
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      // Recurse: switch player and turn.
      int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer), moves[i].first,
                         moves[i].second, false);
      if (Rules::detectCaptureStones(*board, moves[i].first, moves[i].second, currentPlayer))
        board->applyCapture(true);
      delete child;
      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) break;  // Beta cutoff.
    }
    return maxEval;
  } else {
    int minEval = std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); ++i) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer), moves[i].first,
                         moves[i].second, true);
      if (Rules::detectCaptureStones(*board, moves[i].first, moves[i].second, currentPlayer))
        board->applyCapture(true);
      delete child;
      minEval = std::min(minEval, eval);
      beta = std::min(beta, eval);
      if (beta <= alpha) break;  // Alpha cutoff.
    }
    return minEval;
  }
}

std::pair<int, int> getBestMove(Board *board, int depth) {
  int bestScore = std::numeric_limits<int>::min();
  std::pair<int, int> bestMove = std::make_pair(-1, -1);
  std::cout << "next player: " << board->getNextPlayer() << std::endl;
  int currentPlayer = board->getNextPlayer();
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) return bestMove;

  for (size_t i = 0; i < moves.size(); ++i) {
    Board *child = Board::cloneBoard(board);

    // Evaluate the board immediately after applying move (and potential capture).
    int immediateScore =
        Evaluation::evaluatePosition(child, currentPlayer, moves[i].first, moves[i].second);
    // Apply the move for currentPlayer.
    child->setValueBit(moves[i].first, moves[i].second, currentPlayer);

    // Check and apply capture if it occurs.
    if (Rules::detectCaptureStones(*child, moves[i].first, moves[i].second, currentPlayer)) {
      child->applyCapture(true);
    }

    // Use minimax for deeper evaluation (switching to opponent's turn).
    int minimaxScore =
        minimax(child, depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(),
                OPPONENT(currentPlayer), moves[i].first, moves[i].second, false);

    int totalScore = immediateScore + minimaxScore;

    // std::cout << "Move: " << Board::convertIndexToCoordinates(moves[i].first, moves[i].second)
    //           << " Immediate: " << immediateScore << " Minimax: " << minimaxScore
    //           << " Total: " << totalScore << std::endl;

    delete child;

    if (totalScore > bestScore) {
      bestScore = totalScore;
      bestMove = moves[i];
    }
  }
  return bestMove;
}

void simulateAIBattle(Board *pBoard, int searchDepth, int numTurns) {
  for (int turn = 0; turn < numTurns; ++turn) {
    int currentPlayer = pBoard->getNextPlayer();
    std::pair<int, int> bestMove = getBestMove(pBoard, searchDepth);
    if (bestMove.first == -1 || bestMove.second == -1) {
      std::cout << "No valid moves available. Ending simulation." << std::endl;
      break;
    }
    pBoard->setValueBit(bestMove.first, bestMove.second, currentPlayer);
    if (Rules::detectCaptureStones(*pBoard, bestMove.first, bestMove.second, currentPlayer))
      pBoard->applyCapture(true);
    pBoard->switchTurn();  // Update turn after move is made.
    std::cout << "Turn " << turn + 1 << ": Player " << currentPlayer << " moves at ("
              << bestMove.first << ", " << bestMove.second << ")" << std::endl;
    // Optionally, print the board state if your Board class provides a print method.
    pBoard->printBitboard();
  }
}

}  // namespace Minimax
