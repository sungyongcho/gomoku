#include "Minimax.hpp"

#include <cstdlib>
#include <ctime>
#include <limits>
#include <sstream>

namespace Minimax {

static const uint64_t rowMask = ((uint64_t)1 << BOARD_SIZE) - 1;
static std::pair<int, int> killerMoves[MAX_DEPTH][2];

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
        bool isDoubleThree = Rules::detectDoublethreeBit(*board, col, row, board->getNextPlayer());
        if (!isDoubleThree ||
            Rules::detectCaptureStonesNotStore(*board, col, row, board->getNextPlayer()))
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

void initKillerMoves() {
  for (int d = 0; d < MAX_DEPTH; ++d) {
    killerMoves[d][0] = std::make_pair(-1, -1);
    killerMoves[d][1] = std::make_pair(-1, -1);
  }
}

// Helper: check if a move is a killer move for a given depth.
bool isKillerMove(int depth, const std::pair<int, int> &move) {
  return (killerMoves[depth][0] == move || killerMoves[depth][1] == move);
}

struct MoveComparatorMax {
  Board *board;
  int player;
  int depth;  // Current search depth
  MoveComparatorMax(Board *b, int p, int d) : board(b), player(p), depth(d) {}

  bool operator()(const std::pair<int, int> &m1, const std::pair<int, int> &m2) const {
    // Give a bonus if the move is a killer move.
    bool m1_is_killer = false;
    bool m2_is_killer = false;
    // if (depth >= 0 && depth < MAX_DEPTH_LIMIT) { // Add appropriate depth check
    m1_is_killer = (m1 == killerMoves[depth][0] || m1 == killerMoves[depth][1]);
    m2_is_killer = (m2 == killerMoves[depth][0] || m2 == killerMoves[depth][1]);
    // }

    if (m1_is_killer && !m2_is_killer) {
      return true;  // m1 is killer, m2 is not -> m1 first
    }
    if (!m1_is_killer && m2_is_killer) {
      return false;  // m2 is killer, m1 is not -> m2 first
    }

    return Evaluation::evaluatePositionHard(board, player, m1.first, m1.second) >
           Evaluation::evaluatePositionHard(board, player, m2.first, m2.second);
  }
};

struct MoveComparatorMin {
  Board *board;
  int player;
  int depth;  // Current search depth
  MoveComparatorMin(Board *b, int p, int d) : board(b), player(p), depth(d) {}

  bool operator()(const std::pair<int, int> &m1, const std::pair<int, int> &m2) const {
    // Give a bonus if the move is a killer move.
    bool m1_is_killer = false;
    bool m2_is_killer = false;
    // if (depth >= 0 && depth < MAX_DEPTH_LIMIT) { // Add appropriate depth check
    m1_is_killer = (m1 == killerMoves[depth][0] || m1 == killerMoves[depth][1]);
    m2_is_killer = (m2 == killerMoves[depth][0] || m2 == killerMoves[depth][1]);
    // }

    if (m1_is_killer && !m2_is_killer) {
      return true;  // m1 is killer, m2 is not -> m1 first
    }
    if (!m1_is_killer && m2_is_killer) {
      return false;  // m2 is killer, m1 is not -> m2 first
    }

    return Evaluation::evaluatePositionHard(board, player, m1.first, m1.second) <
           Evaluation::evaluatePositionHard(board, player, m2.first, m2.second);
  }
};

// int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int
// lastY,
//             bool isMaximizing) {
//   // Terminal condition: if we've reached the maximum search depth.
//   if (depth == 0) {
//     // Evaluate board based on the last move (played by opponent of currentPlayer)
//     return Evaluation::evaluatePositionHard(board, OPPONENT(currentPlayer), lastX, lastY);
//   }

//   // Generate candidate moves.
//   std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
//   if (moves.empty())
//     return Evaluation::evaluatePositionHard(board, OPPONENT(currentPlayer), lastX, lastY);

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
//       int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer),
//       moves[i].first,
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
//       int eval = minimax(child, depth - 1, alpha, beta, OPPONENT(currentPlayer),
//       moves[i].first,
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

int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY,
            bool isMaximizing) {
  // Terminal condition: if we've reached maximum depth.
  if (depth == 0 || Rules::isWinningMove(board, currentPlayer, lastX, lastY)) {
    // board->printBitboard();
    int eval = Evaluation::evaluatePositionHard(board, currentPlayer, lastX, lastY);
    return eval;
  }

  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    int eval = Evaluation::evaluatePositionHard(board, currentPlayer, lastX, lastY);
    return eval;
  }

  int bestEval;

  if (isMaximizing) {
    // Use the new comparator that takes depth.
    bestEval = std::numeric_limits<int>::min();
    for (size_t i = 0; i < moves.size(); ++i) {
      UndoInfo info = board->makeMove(moves[i].first, moves[i].second);
      int playerMakingMove = board->getNextPlayer();  // Get player before making move
      // childMax->setValueBit(moves[i].first, moves[i].second, currentPlayer);
      int eval = minimax(board, depth - 1, alpha, beta, playerMakingMove, moves[i].first,
                         moves[i].second, false);
      board->undoMove(info);
      if (eval > bestEval) {
        bestEval = eval;
      }
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        // On cutoff, record the move as a killer move for this depth if not already recorded.
        if (killerMoves[depth][0] != moves[i] && killerMoves[depth][1] != moves[i]) {
          killerMoves[depth][1] = killerMoves[depth][0];
          killerMoves[depth][0] = moves[i];
        }
        break;  // Beta cutoff.
      }
    }
  } else {
    bestEval = std::numeric_limits<int>::max();
    for (size_t i = 0; i < moves.size(); ++i) {
      UndoInfo info = board->makeMove(moves[i].first, moves[i].second);
      int playerMakingMove = board->getNextPlayer();  // Get player before making move
      int eval = minimax(board, depth - 1, alpha, beta, playerMakingMove, moves[i].first,
                         moves[i].second, true);
      board->undoMove(info);
      if (eval < bestEval) {
        bestEval = eval;
      }
      beta = std::min(beta, eval);
      if (beta <= alpha) {
        // On cutoff, record the move as a killer move for this depth if not already recorded.
        if (killerMoves[depth][0] != moves[i] && killerMoves[depth][1] != moves[i]) {
          killerMoves[depth][1] = killerMoves[depth][0];
          killerMoves[depth][0] = moves[i];
        }
        break;  // Alpha cutoff.
      }
    }
  }

  return bestEval;
}

std::pair<int, int> getBestMove(Board *board, int depth) {
  int bestScore = std::numeric_limits<int>::min();
  std::pair<int, int> bestMove = std::make_pair(-1, -1);
  int currentPlayer = board->getNextPlayer();
  int root_alpha = std::numeric_limits<int>::min();  // Initial alpha = -infinity
  int root_beta = std::numeric_limits<int>::max();   // Initial beta = +infinity
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) return bestMove;

  // Order moves for the maximizing player.
  MoveComparatorMax cmp(board, board->getNextPlayer(), depth);
  std::sort(moves.begin(), moves.end(), cmp);

  std::cout << "depth: " << depth << " player" << currentPlayer << std::endl;
  // Evaluate each candidate move.
  for (size_t i = 0; i < moves.size(); ++i) {
    if (Evaluation::evaluatePositionHard(board, currentPlayer, moves[i].first, moves[i].second) >
        MINIMAX_TERMINATION)
      return moves[i];
    int playerMakingMove = board->getNextPlayer();  // Get player before making move

    UndoInfo info = board->makeMove(moves[i].first, moves[i].second);
    int score = minimax(board, depth - 1, root_alpha, root_beta, playerMakingMove, moves[i].first,
                        moves[i].second, false);
    board->undoMove(info);
    if (score > bestScore) {
      bestScore = score;
      bestMove = moves[i];
    }
  }
  std::cout << "final: " << board->getNextPlayer() << std::endl;
  return bestMove;
}

std::pair<int, int> getBestMovePV(Board *board, int depth, std::pair<int, int> &pvMove) {
  int currentPlayer = board->getNextPlayer();
  std::cout << "next player" << currentPlayer << std::endl;
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) return std::make_pair(-1, -1);

  // If a PV move exists and is valid, bring it to the front.
  if (pvMove.first != -1 && pvMove.second != -1) {
    std::vector<std::pair<int, int> >::iterator it = std::find(moves.begin(), moves.end(), pvMove);
    if (it != moves.end()) {
      std::iter_swap(moves.begin(), it);
    }
  }

  // Now sort the candidate moves using your comparator.
  MoveComparatorMax cmp(board, currentPlayer, depth);
  std::sort(moves.begin(), moves.end(), cmp);

  // Evaluate each candidate move.
  int bestScore = std::numeric_limits<int>::min();
  std::pair<int, int> bestMove = std::make_pair(-1, -1);
  for (size_t i = 0; i < moves.size(); ++i) {
    Board *child = new Board(*board);

    // int immediateScore =
    //     Evaluation::evaluatePositionHard(child, currentPlayer, moves[i].first,
    //     moves[i].second);

    // // Apply the move.
    // child->setValueBit(moves[i].first, moves[i].second, currentPlayer);
    // if (Rules::detectCaptureStones(*child, moves[i].first, moves[i].second, currentPlayer))
    //   child->applyCapture(true);
    // child->switchTurn();
    // Use your minimax function (with transposition table and iterative deepening) for deeper
    // evaluation.
    int minimaxScore =
        minimax(child, depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(),
                OPPONENT(currentPlayer), moves[i].first, moves[i].second, false);

    delete child;

    if (minimaxScore > bestScore) {
      bestScore = minimaxScore;
      bestMove = moves[i];
    }
  }

  // Update the PV move for future iterations.
  pvMove = bestMove;
  return bestMove;
}

std::pair<int, int> iterativeDeepening(Board *board, int maxDepth, double timeLimitSeconds) {
  std::pair<int, int> bestMove = std::make_pair(-1, -1);
  std::pair<int, int> pvMove = std::make_pair(-1, -1);  // Initially no PV move.
  std::clock_t startTime = std::clock();

  // board->setLastEvalScore(Evaluation::evaluatePositionHard(board, board->getLastPlayer(),
  //                                                          board->getLastX(),
  //                                                          board->getLastY()));

  // Optionally, clear the transposition table.
  transTable.clear();
  initKillerMoves();

  for (int depth = 1; depth <= maxDepth; ++depth) {
    // Check elapsed time.
    double elapsedTime = static_cast<double>(std::clock() - startTime) / CLOCKS_PER_SEC;
    if (elapsedTime > timeLimitSeconds) {
      // Time limit reached; break out.
      break;
    }
    bestMove = getBestMovePV(board, depth, pvMove);
    // bestMove = getBestMove(board, depth);
    // Optionally print debug info:
    // std::cout << "Depth " << depth << " best move: (" << bestMove.first << ", "
    //           << bestMove.second << ") - Elapsed: " << elapsedTime << " s\n";
  }
  return bestMove;
}

}  // namespace Minimax
