#include "Minimax.hpp"

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
      if (candidates & (1ULL << col)) {
        if (!Rules::detectDoublethreeBit(*board, col, row, board->getNextPlayer()))
          moves.push_back(std::make_pair(col, row));
      }
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

// int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int
// lastY,
//             bool isMaximizing) {
//   // Terminal condition: if we've reached the maximum search depth.
//   if (depth == 0) {
//     // Evaluate board based on the last move (played by opponent of currentPlayer)
//     return Evaluation::evaluatePosition(board, OPPONENT(currentPlayer), lastX, lastY);
//   }

//   // Generate candidate moves.
//   std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
//   if (moves.empty())
//     return Evaluation::evaluatePosition(board, OPPONENT(currentPlayer), lastX, lastY);

//   if (isMaximizing) {
//     MoveComparatorMax cmp(board, currentPlayer);
//     std::sort(moves.begin(), moves.end(), cmp);
//   } else {
//     MoveComparatorMin cmp(board, currentPlayer);
//     std::sort(moves.begin(), moves.end(), cmp);
//   }

//   // if (moves.size() > 1) {
//   //   moves.resize(moves.size() / 2);
//   // }

//   if (isMaximizing) {
//     int maxEval = std::numeric_limits<int>::min();
//     for (size_t i = 0; i < moves.size(); ++i) {
//       // Create a child board state.
//       Board *child = Board::cloneBoard(board);
//       child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
//       // Recurse: switch player and turn.
//       int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer), moves[i].first,
//                          moves[i].second, false);
//       if (Rules::detectCaptureStones(*board, moves[i].first, moves[i].second, currentPlayer))
//         board->applyCapture(true);
//       delete child;
//       maxEval = std::max(maxEval, eval);
//       alpha = std::max(alpha, eval);
//       if (beta <= alpha) break;  // Beta cutoff.
//     }
//     return maxEval;
//   } else {
//     int minEval = std::numeric_limits<int>::max();
//     for (size_t i = 0; i < moves.size(); ++i) {
//       Board *child = Board::cloneBoard(board);
//       child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
//       int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer), moves[i].first,
//                          moves[i].second, true);
//       if (Rules::detectCaptureStones(*board, moves[i].first, moves[i].second, currentPlayer))
//         board->applyCapture(true);
//       delete child;
//       minEval = std::min(minEval, eval);
//       beta = std::min(beta, eval);
//       if (beta <= alpha) break;  // Alpha cutoff.
//     }
//     return minEval;
//   }
// }

// std::pair<int, int> getBestMove(Board *board, int depth) {
//   int bestScore = std::numeric_limits<int>::min();
//   std::pair<int, int> bestMove = std::make_pair(-1, -1);
//   int currentPlayer = board->getNextPlayer();
//   std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
//   if (moves.empty()) return bestMove;

//   MoveComparatorMax cmp(board, currentPlayer);
//   std::sort(moves.begin(), moves.end(), cmp);

//   // if (moves.size() > 1) {
//   //   moves.resize(moves.size() / 2);
//   // }

//   for (size_t i = 0; i < moves.size(); ++i) {
//     Board *child = Board::cloneBoard(board);

//     // Evaluate the board immediately after applying move (and potential capture).
//     int immediateScore =
//         Evaluation::evaluatePosition(child, currentPlayer, moves[i].first, moves[i].second);
//     // Apply the move for currentPlayer.
//     child->setValueBit(moves[i].first, moves[i].second, currentPlayer);

//     // Check and apply capture if it occurs.
//     if (Rules::detectCaptureStones(*child, moves[i].first, moves[i].second, currentPlayer)) {
//       child->applyCapture(true);
//     }

//     // Use minimax for deeper evaluation (switching to opponent's turn).
//     int minimaxScore =
//         minimax(child, depth - 1, std::numeric_limits<int>::min(),
//         std::numeric_limits<int>::max(),
//                 OPPONENT(currentPlayer), moves[i].first, moves[i].second, false);

//     int totalScore = immediateScore + minimaxScore;

//     // std::cout << "Move: " << Board::convertIndexToCoordinates(moves[i].first, moves[i].second)
//     //           << " Immediate: " << immediateScore << " Minimax: " << minimaxScore
//     //           << " Total: " << totalScore << std::endl;

//     delete child;

//     if (totalScore > bestScore) {
//       bestScore = totalScore;
//       bestMove = moves[i];
//     }
//   }
//   return bestMove;
// }

int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY,
            bool isMaximizing) {
  // Compute the current board's hash.
  uint64_t hash = board->getHash();

  // Check the transposition table.
  std::map<uint64_t, TTEntry>::iterator it = transTable.find(hash);
  if (it != transTable.end() && it->second.depth >= depth) {
    TTEntry entry = it->second;
    if (entry.flag == EXACT)
      return entry.score;
    else if (entry.flag == LOWERBOUND)
      alpha = std::max(alpha, entry.score);
    else if (entry.flag == UPPERBOUND)
      beta = std::min(beta, entry.score);
    if (alpha >= beta) return entry.score;
  }

  // Terminal condition: if we've reached maximum depth.
  if (depth == 0) {
    int eval = Evaluation::evaluatePosition(board, OPPONENT(currentPlayer), lastX, lastY);
    TTEntry entry = {eval, depth, std::make_pair(-1, -1), EXACT};
    transTable[hash] = entry;
    return eval;
  }

  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    int eval = Evaluation::evaluatePosition(board, OPPONENT(currentPlayer), lastX, lastY);
    TTEntry entry = {eval, depth, std::make_pair(-1, -1), EXACT};
    transTable[hash] = entry;
    return eval;
  }

  int bestEval;
  // Save original alpha and beta for later bound determination.
  int originalAlpha = alpha;
  int originalBeta = beta;
  std::pair<int, int> bestMove = std::make_pair(-1, -1);

  if (isMaximizing) {
    MoveComparatorMax cmp(board, currentPlayer);
    std::sort(moves.begin(), moves.end(), cmp);
    bestEval = std::numeric_limits<int>::min();
    for (size_t i = 0; i < moves.size(); ++i) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer), moves[i].first,
                         moves[i].second, false);
      delete child;
      if (eval > bestEval) {
        bestEval = eval;
        bestMove = moves[i];
      }
      alpha = std::max(alpha, eval);
      if (beta <= alpha) break;  // Beta cutoff.
    }
  } else {
    MoveComparatorMin cmp(board, currentPlayer);
    std::sort(moves.begin(), moves.end(), cmp);
    bestEval = std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); ++i) {
      Board *child = Board::cloneBoard(board);
      child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer), moves[i].first,
                         moves[i].second, true);
      delete child;
      if (eval < bestEval) {
        bestEval = eval;
        bestMove = moves[i];
      }
      beta = std::min(beta, eval);
      if (beta <= alpha) break;  // Alpha cutoff.
    }
  }

  // Determine the bound type for the entry.
  BoundType flag;
  if (bestEval <= originalAlpha)
    flag = UPPERBOUND;
  else if (bestEval >= originalBeta)
    flag = LOWERBOUND;
  else
    flag = EXACT;

  // Store the computed value in the transposition table.
  TTEntry newEntry = {bestEval, depth, bestMove, flag};
  transTable[hash] = newEntry;
  return bestEval;
}

std::pair<int, int> getBestMove(Board *board, int depth) {
  int bestScore = std::numeric_limits<int>::min();
  std::pair<int, int> bestMove = std::make_pair(-1, -1);
  int currentPlayer = board->getNextPlayer();
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) return bestMove;

  // Order moves for the maximizing player.
  MoveComparatorMax cmp(board, currentPlayer);
  std::sort(moves.begin(), moves.end(), cmp);

  // Evaluate each candidate move.
  for (size_t i = 0; i < moves.size(); ++i) {
    Board *child = Board::cloneBoard(board);

    // Optionally, get an immediate evaluation (if useful).
    int immediateScore =
        Evaluation::evaluatePosition(child, currentPlayer, moves[i].first, moves[i].second);

    // Apply the move.
    child->setValueBit(moves[i].first, moves[i].second, currentPlayer);

    // Check and apply captures.
    if (Rules::detectCaptureStones(*child, moves[i].first, moves[i].second, currentPlayer))
      child->applyCapture(true);

    // Use minimax for deeper evaluation (switching to opponent's turn).
    int minimaxScore =
        minimax(child, depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(),
                OPPONENT(currentPlayer), moves[i].first, moves[i].second, false);

    delete child;

    int totalScore = immediateScore + minimaxScore;
    if (totalScore > bestScore) {
      bestScore = totalScore;
      bestMove = moves[i];
    }
  }
  return bestMove;
}

std::pair<int, int> iterativeDeepening(Board *board, int maxDepth, double timeLimitSeconds) {
  std::pair<int, int> bestMove = std::make_pair(-1, -1);
  std::clock_t startTime = std::clock();

  // Optionally clear the transposition table before starting.
  transTable.clear();

  for (int depth = 1; depth <= maxDepth; ++depth) {
    // Check elapsed time.
    double elapsedTime = static_cast<double>(std::clock() - startTime) / CLOCKS_PER_SEC;
    if (elapsedTime > timeLimitSeconds) {
      std::cout << "Time limit reached at depth " << depth << ".\n";
      break;
    }
    bestMove = getBestMove(board, depth);

    std::cout << "Depth " << depth << " best move: (" << bestMove.first << ", " << bestMove.second
              << ") - Elapsed: " << elapsedTime << " s\n";
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
