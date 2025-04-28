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
        bool enableDoubleThreeRestriction = board->getEnableDoubleThreeRestriction();
        bool isDoubleThree =
            enableDoubleThreeRestriction
                ? Rules::detectDoublethreeBit(*board, col, row, board->getNextPlayer())
                : false;
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

Evaluation::EvaluationEntry quiescenceSearch(Board *board, Evaluation::EvaluationEntry alpha,
                                             Evaluation::EvaluationEntry beta, bool isMaximizing) {
  // 1. Evaluate Stand-Pat Score
  //    Perspective is crucial. Evaluate from the point of view of the player whose turn it is.
  int playerWhoseTurnItIs = board->getNextPlayer();
  // Use -1,-1 or appropriate dummy coords if last move isn't relevant here
  Evaluation::EvaluationEntry stand_pat =
      Evaluation::evaluatePositionHard(board, playerWhoseTurnItIs, -1, -1);
  // 2. Initial Pruning based on Stand-Pat
  if (isMaximizing) {
    if (stand_pat >= beta) {
      return beta;  // Fail-hard beta cutoff
    }
    alpha = std::max(alpha, stand_pat);
  } else {  // Minimizing
    if (stand_pat <= alpha) {
      return alpha;  // Fail-hard alpha cutoff
    }
    beta = std::min(beta, stand_pat);
  }

  // 3. Generate Only Capture Moves
  std::vector<std::pair<int, int> > captureMoves = generateCandidateMoves(board);

  // 4. Base Case: No captures means position is quiet
  if (captureMoves.empty()) {
    return stand_pat;
  }

  // --- Optional: Order capture moves (e.g., most valuable first) ---

  // 5. Explore Capture Moves
  Evaluation::EvaluationEntry bestEval = stand_pat;  // Initialize with stand-pat

  for (size_t i = 0; i < captureMoves.size(); ++i) {
    UndoInfo info = board->makeMove(captureMoves[i].first, captureMoves[i].second);
    // Recursively call quiescence search for the opponent
    Evaluation::EvaluationEntry eval = quiescenceSearch(board, alpha, beta, !isMaximizing);
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

Evaluation::EvaluationEntry minimax(Board *board, int depth, Evaluation::EvaluationEntry alpha,
                                    Evaluation::EvaluationEntry beta, int currentPlayer, int lastX,
                                    int lastY, bool isMaximizing) {
  // --- [Win check, depth check (calling quiescenceSearch), etc. as before] ---
  int playerWhoJustMoved = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  Evaluation::EvaluationEntry evalScore =
      Evaluation::evaluatePositionHard(board, playerWhoJustMoved, lastX, lastY);
  if (lastX != -1 && evalScore.score >= MINIMAX_TERMINATION) {
    if (board->getCapturedStones().size() > 0) {
      board->applyCapture(true);
    }
    return evalScore;
  }

  if (depth == 0) {
    if (board->getCapturedStones().size() > 0) {
      board->applyCapture(true);
    }
    return evalScore;
  }

  if (board->getCapturedStones().size() > 0) {
    board->applyCapture(true);
  }

  // const int NULL_MOVE_REDUCTION = 2;
  // // Preconditions: Check if NMP is applicable
  // // - Depth must be high enough (e.g., depth >= NULL_MOVE_REDUCTION + 1)
  // // - Not in quiescence search (this function isn't QSearch, so okay)
  // // - Avoid near end-game? (Less critical in Gomoku perhaps)
  // // - Not root node? (Sometimes NMP is skipped at root or near-root)
  // if (depth >= (NULL_MOVE_REDUCTION + 1) /* && other conditions if needed */) {
  //   // 1. Make Null Move (conceptually switch player)
  //   // board->switchPlayer(); // Or however you handle turn switching
  //   // uint64_t originalHash = board->getHash(); // If hash includes side-to-move
  //   // board->updateHashForSideToMove();

  //   // 2. Recursive Call with reduced depth and swapped bounds
  //   // Note: Pass the *opponent's* perspective for alpha/beta
  //   int null_score = -minimax(board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1,
  //                             (currentPlayer == PLAYER_1 ? PLAYER_2 : PLAYER_1), -1,
  //                             -1,              // No real last move for null move
  //                             !isMaximizing);  // Flip maximizing flag

  //   // 3. Undo Null Move
  //   // board->switchPlayer();
  //   // board->setHash(originalHash); // Restore hash if needed

  //   // 4. Check result and prune if null_score >= beta
  //   if (null_score >= beta) {
  //     // Null move indicates the position is strong enough to likely cause a cutoff
  //     return beta;  // Return beta (fail-hard)
  //   }
  // }

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
    Evaluation::EvaluationEntry move_score =
        Evaluation::evaluatePositionHard(board, currentPlayer, m.first, m.second);
    // Check killer status (ensure 'm' can be compared to killerMoves elements)
    bool is_killer = false;
    if (depth >= 0 /* && depth < MAX_DEPTH_LIMIT */) {  // Add bounds check if necessary
      is_killer = (m == killerMoves[depth][0] || m == killerMoves[depth][1]);
    }
    // Add to vector using push_back and explicit constructor call
    scored_moves.push_back(ScoredMove(move_score.score, m, is_killer));
  }

  // 3. Sort the scored_moves vector using the appropriate functor
  if (isMaximizing) {
    std::sort(scored_moves.begin(), scored_moves.end(), CompareScoredMovesMax());
  } else {  // Minimizing
    std::sort(scored_moves.begin(), scored_moves.end(), CompareScoredMovesMin());
  }

  // ***** END OF MOVE ORDERING *****

  Evaluation::EvaluationEntry bestEval;

  // 4. Iterate through the *sorted* scored_moves (using iterators)
  if (isMaximizing) {
    bestEval.score = std::numeric_limits<int>::min();
    for (std::vector<ScoredMove>::const_iterator it = scored_moves.begin();
         it != scored_moves.end(); ++it) {
      const ScoredMove &scored_move = *it;                 // Explicit type
      const std::pair<int, int> &move = scored_move.move;  // Get the actual move pair

      UndoInfo info = board->makeMove(move.first, move.second);
      int nextPlayer = board->getNextPlayer();
      Evaluation::EvaluationEntry eval =
          minimax(board, depth - 1, alpha, beta, nextPlayer, move.first, move.second, false);
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
    bestEval.score = std::numeric_limits<int>::max();
    for (std::vector<ScoredMove>::const_iterator it = scored_moves.begin();
         it != scored_moves.end(); ++it) {
      const ScoredMove &scored_move = *it;                 // Explicit type
      const std::pair<int, int> &move = scored_move.move;  // Get the actual move pair

      UndoInfo info = board->makeMove(move.first, move.second);
      int nextPlayer = board->getNextPlayer();
      Evaluation::EvaluationEntry eval =
          minimax(board, depth - 1, alpha, beta, nextPlayer, move.first, move.second, true);
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

  // std::cout << "BESTEVAL DEPTH " << currentPlayer << " (" << depth << ") " << lastX << "," <<
  // lastY
  //           << std::endl;
  Evaluation::printEvalEntry(bestEval);
  return bestEval;
}

std::pair<int, int> getBestMove(Board *board, int depth) {
  int bestScore = std::numeric_limits<int>::min();
  std::pair<int, int> bestMove = std::make_pair(-1, -1);

  int currentPlayer = board->getNextPlayer();
  Evaluation::EvaluationEntry root_alpha;
  Evaluation::EvaluationEntry root_beta;

  root_alpha.score = std::numeric_limits<int>::min();  // Initial alpha = -infinity
  root_beta.score = std::numeric_limits<int>::max();   // Initial beta = +infinity

  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    return bestMove;
  }

  initKillerMoves();
  // Order moves for the maximizing player.
  // ***** APPLY PRE-CALCULATION & SORTING HERE *****
  std::vector<ScoredMove> scored_moves;
  scored_moves.reserve(moves.size());  // Good practice, C++98 compatible

  // 2. Calculate score ONCE for each move (using iterators)
  for (std::vector<std::pair<int, int> >::const_iterator it = moves.begin(); it != moves.end();
       ++it) {
    const std::pair<int, int> &m = *it;  // Explicit type for the move
    Evaluation::EvaluationEntry move_eval =
        Evaluation::evaluatePositionHard(board, currentPlayer, m.first, m.second);
    // Check killer status (ensure 'm' can be compared to killerMoves elements)
    bool is_killer = false;
    if (depth >= 0 /* && depth < MAX_DEPTH_LIMIT */) {  // Add bounds check if necessary
      is_killer = (m == killerMoves[depth][0] || m == killerMoves[depth][1]);
    }
    // Add to vector using push_back and explicit constructor call
    scored_moves.push_back(ScoredMove(move_eval.score, m, is_killer));
  }

  std::sort(scored_moves.begin(), scored_moves.end(), CompareScoredMovesMax());
  // ***** END PRE-CALCULATION & SORTING *****

  std::cout << "depth: " << depth << " player" << currentPlayer << std::endl;

  Evaluation::EvaluationEntry bestEval;
  // Evaluate each candidate move.
  for (size_t i = 0; i < scored_moves.size(); ++i) {
    if (scored_moves[i].score > MINIMAX_TERMINATION) {
      return scored_moves[i].move;
    }
    int playerMakingMove = board->getNextPlayer();  // Get player before making move

    UndoInfo info = board->makeMove(scored_moves[i].move.first, scored_moves[i].move.second);
    bestEval = minimax(board, depth - 1, root_alpha, root_beta, playerMakingMove,
                       scored_moves[i].move.first, scored_moves[i].move.second, false);
    board->undoMove(info);
    if (bestEval.score > bestScore) {
      bestScore = bestEval.score;
      bestMove = scored_moves[i].move;
      root_alpha = std::max(root_alpha, bestEval);
    }
  }
  std::cout << "final: " << board->getNextPlayer() << std::endl;
  std::cout << "getBestMove position: " << bestMove.first << ", " << bestMove.second << std::endl;
  std::cout << "getBestMove score: " << bestScore << std::endl;
  std::cout << "score" << board->getNextPlayerScore() << std::endl;
  std::cout << "score" << board->getLastPlayerScore() << std::endl;
  Evaluation::printEvalEntry(bestEval);

  return bestMove;
}

}  // namespace Minimax
