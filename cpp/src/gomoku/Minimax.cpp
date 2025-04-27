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

std::vector<std::pair<int, int> > generateCaptureMoves(Board *&board) {
  std::vector<std::pair<int, int> > moves;
  uint64_t occupancy[BOARD_SIZE];
  uint64_t neighbor[BOARD_SIZE];

  board->getOccupancy(occupancy);
  computeNeighborMask(occupancy, neighbor);

  for (int row = 0; row < BOARD_SIZE; row++) {
    uint64_t candidates = neighbor[row] & (~occupancy[row]) & rowMask;
    for (int col = 0; col < BOARD_SIZE; col++) {
      if (candidates & (1ULL << col)) {
        if (Rules::detectCaptureStonesNotStore(*board, col, row, board->getNextPlayer()))
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

// Helper structure to store move, its score, and killer status
struct ScoredMove {
  int score;
  std::pair<int, int> move;
  bool is_killer;

  // Constructor (C++98 style)
  ScoredMove(int s, std::pair<int, int> m, bool ik) : score(s), move(m), is_killer(ik) {}
};

// Comparator functor for sorting ScoredMoves for the Maximizing player
struct CompareScoredMovesMax {
  bool operator()(const ScoredMove &a, const ScoredMove &b) const {
    if (a.is_killer && !b.is_killer) return true;  // Killer moves first
    if (!a.is_killer && b.is_killer) return false;
    // If killer status is the same, sort by score descending (best first)
    return a.score > b.score;
  }
};

// Comparator functor for sorting ScoredMoves for the Minimizing player
struct CompareScoredMovesMin {
  bool operator()(const ScoredMove &a, const ScoredMove &b) const {
    if (a.is_killer && !b.is_killer) return true;  // Killer moves first
    if (!a.is_killer && b.is_killer) return false;
    // If killer status is the same, sort by score ascending (best first)
    return a.score < b.score;
  }
};

int quiescenceSearch(Board *board, int alpha, int beta, bool isMaximizing) {
  // 1. Evaluate Stand-Pat Score
  //    Perspective is crucial. Evaluate from the point of view of the player whose turn it is.
  int playerWhoseTurnItIs = board->getNextPlayer();
  // Use -1,-1 or appropriate dummy coords if last move isn't relevant here
  int stand_pat_score = Evaluation::evaluatePositionHard(board, playerWhoseTurnItIs, -1, -1);
  // 2. Initial Pruning based on Stand-Pat
  if (isMaximizing) {
    if (stand_pat_score >= beta) {
      return beta;  // Fail-hard beta cutoff
    }
    alpha = std::max(alpha, stand_pat_score);
  } else {  // Minimizing
    if (stand_pat_score <= alpha) {
      return alpha;  // Fail-hard alpha cutoff
    }
    beta = std::min(beta, stand_pat_score);
  }

  // 3. Generate Only Capture Moves
  std::vector<std::pair<int, int> > captureMoves = generateCandidateMoves(board);

  // 4. Base Case: No captures means position is quiet
  if (captureMoves.empty()) {
    return stand_pat_score;
  }

  // --- Optional: Order capture moves (e.g., most valuable first) ---

  // 5. Explore Capture Moves
  int bestEval = stand_pat_score;  // Initialize with stand-pat

  for (size_t i = 0; i < captureMoves.size(); ++i) {
    UndoInfo info = board->makeMove(captureMoves[i].first, captureMoves[i].second);
    // Recursively call quiescence search for the opponent
    int eval = quiescenceSearch(board, alpha, beta, !isMaximizing);
    board->undoMove(info);

    if (isMaximizing) {
      bestEval = std::max(bestEval, eval);  // Update best score found
      alpha = std::max(alpha, eval);        // Update alpha
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    } else {                                // Minimizing
      bestEval = std::min(bestEval, eval);  // Update best score found
      beta = std::min(beta, eval);          // Update beta
      if (beta <= alpha) {
        break;  // Alpha cutoff
      }
    }
  }
  return bestEval;
}

int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY,
            bool isMaximizing) {
  // --- [Win check, depth check (calling quiescenceSearch), etc. as before] ---
  int playerWhoJustMoved = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  if (lastX != -1 && Rules::isWinningMove(board, playerWhoJustMoved, lastX, lastY)) {
    return Evaluation::evaluatePositionHard(board, currentPlayer, lastX, lastY);
  }
  if (depth == 0) {
    return quiescenceSearch(board, alpha, beta, isMaximizing);
  }

  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    return Evaluation::evaluatePositionHard(board, currentPlayer, lastX, lastY);
  }

  // ***** MOVE ORDERING (C++98 VERSION) *****

  // 1. Create vector for scored moves
  std::vector<ScoredMove> scored_moves;
  scored_moves.reserve(moves.size());  // Good practice, C++98 compatible

  // 2. Calculate score ONCE for each move (using iterators)
  for (std::vector<std::pair<int, int> >::const_iterator it = moves.begin(); it != moves.end();
       ++it) {
    const std::pair<int, int> &m = *it;  // Explicit type for the move
    int move_score = Evaluation::evaluatePositionHard(board, currentPlayer, m.first, m.second);
    // Check killer status (ensure 'm' can be compared to killerMoves elements)
    bool is_killer = false;
    if (depth >= 0 /* && depth < MAX_DEPTH_LIMIT */) {  // Add bounds check if necessary
      is_killer = (m == killerMoves[depth][0] || m == killerMoves[depth][1]);
    }
    // Add to vector using push_back and explicit constructor call
    scored_moves.push_back(ScoredMove(move_score, m, is_killer));
  }

  // 3. Sort the scored_moves vector using the appropriate functor
  if (isMaximizing) {
    std::sort(scored_moves.begin(), scored_moves.end(), CompareScoredMovesMax());
  } else {  // Minimizing
    std::sort(scored_moves.begin(), scored_moves.end(), CompareScoredMovesMin());
  }

  // ***** END OF MOVE ORDERING *****

  int bestEval;

  // 4. Iterate through the *sorted* scored_moves (using iterators)
  if (isMaximizing) {
    bestEval = std::numeric_limits<int>::min();
    for (std::vector<ScoredMove>::const_iterator it = scored_moves.begin();
         it != scored_moves.end(); ++it) {
      const ScoredMove &scored_move = *it;                 // Explicit type
      const std::pair<int, int> &move = scored_move.move;  // Get the actual move pair

      UndoInfo info = board->makeMove(move.first, move.second);
      int nextPlayer = board->getNextPlayer();
      int eval = minimax(board, depth - 1, alpha, beta, nextPlayer, move.first, move.second, false);
      board->undoMove(info);

      bestEval = std::max(bestEval, eval);
      alpha = std::max(alpha, eval);

      if (beta <= alpha) {
        // Store killer move (use 'move', the std::pair)
        if (killerMoves[depth][0] != move && killerMoves[depth][1] != move) {
          killerMoves[depth][1] = killerMoves[depth][0];
          killerMoves[depth][0] = move;
        }
        break;  // Beta cutoff.
      }
    }
  } else {  // Minimizing
    bestEval = std::numeric_limits<int>::max();
    for (std::vector<ScoredMove>::const_iterator it = scored_moves.begin();
         it != scored_moves.end(); ++it) {
      const ScoredMove &scored_move = *it;                 // Explicit type
      const std::pair<int, int> &move = scored_move.move;  // Get the actual move pair

      UndoInfo info = board->makeMove(move.first, move.second);
      int nextPlayer = board->getNextPlayer();
      int eval = minimax(board, depth - 1, alpha, beta, nextPlayer, move.first, move.second, true);
      board->undoMove(info);

      bestEval = std::min(bestEval, eval);
      beta = std::min(beta, eval);

      if (beta <= alpha) {
        // Store killer move (use 'move', the std::pair)
        if (killerMoves[depth][0] != move && killerMoves[depth][1] != move) {
          killerMoves[depth][1] = killerMoves[depth][0];
          killerMoves[depth][0] = move;
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
  initKillerMoves();
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
      root_alpha = std::max(root_alpha, score);
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
