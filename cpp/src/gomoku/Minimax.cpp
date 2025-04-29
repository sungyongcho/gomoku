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

// Generate candidate moves using row-based neighbor mask.
std::vector<std::pair<int, int> > generateCriticalMoves(Board *&board, int player) {
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
        if ((!isDoubleThree ||
             Rules::detectCaptureStonesNotStore(*board, col, row, board->getNextPlayer())) &&
            Rules::isWinningMove(board, player, col, row))
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

int quiescenceSearch(Board *board, int alpha, int beta, bool isMaximizing, int x, int y) {
  // 1. Evaluate Stand-Pat Score
  //    Perspective is crucial. Evaluate from the point of view of the player whose turn it is.
  int playerWhoseTurnItIs = board->getNextPlayer();
  // Use -1,-1 or appropriate dummy coords if last move isn't relevant here
  int stand_pat_score = Evaluation::evaluatePositionHard(board, playerWhoseTurnItIs, x, y);
  // 2. Initial Pruning based on Stand-Pat
  if (isMaximizing) {
    if (stand_pat_score >= beta) {
      if (board->getCapturedStones().size() > 0) {
        board->applyCapture(true);
      }
      return beta;  // Fail-hard beta cutoff
    }
    alpha = std::max(alpha, stand_pat_score);
  } else {  // Minimizing
    if (stand_pat_score <= alpha) {
      if (board->getCapturedStones().size() > 0) {
        board->applyCapture(true);
      }
      return alpha;  // Fail-hard alpha cutoff
    }
    beta = std::min(beta, stand_pat_score);
  }

  if (board->getCapturedStones().size() > 0) {
    board->applyCapture(true);
  }

  // 3. Generate Only Capture Moves
  std::vector<std::pair<int, int> > captureMoves = generateCaptureMoves(board);

  // 4. Base Case: No captures means position is quiet
  if (captureMoves.empty()) {
    return stand_pat_score;
  }

  int bestEval = stand_pat_score;  // Initialize with stand-pat

  for (size_t i = 0; i < captureMoves.size(); ++i) {
    UndoInfo info = board->makeMove(captureMoves[i].first, captureMoves[i].second);
    // Recursively call quiescence search for the opponent
    int eval = quiescenceSearch(board, alpha, beta, !isMaximizing, captureMoves[i].first,
                                captureMoves[i].second);
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
  // --- Alpha-Beta Preamble ---
  int initial_alpha = alpha;                // Store original alpha for TT storing logic later
  uint64_t currentHash = board->getHash();  // *** Requires Board::getHash() ***
  std::pair<int, int> bestMoveFromTT = std::make_pair(-1, -1);  // Candidate Hash Move from TT

  boost::unordered_map<uint64_t, TTEntry>::iterator tt_it = transTable.find(currentHash);
  if (tt_it != transTable.end()) {
    const TTEntry &entry = tt_it->second;  // Use const reference
    bestMoveFromTT = entry.bestMove;       // Get potential hash move for ordering

    // Check if stored depth is sufficient to use the score/bound
    if (entry.depth >= depth) {
      int stored_score = entry.score;
      if (entry.flag == EXACT) {
        if (board->getCapturedStones().size() > 0) {
          board->applyCapture(true);
        }
        return stored_score;  // Exact score found, return immediately
      } else if (entry.flag == LOWERBOUND) {
        // Stored score is a lower bound, update our alpha
        alpha = std::max(alpha, stored_score);
      } else if (entry.flag == UPPERBOUND) {
        // Stored score is an upper bound, update our beta
        beta = std::min(beta, stored_score);
      }
      // Check for cutoff after potentially updating bounds from TT
      if (alpha >= beta) {
        // The stored information caused a cutoff
        if (board->getCapturedStones().size() > 0) {
          board->applyCapture(true);
        }
        return stored_score;  // Return the score that caused the cutoff
      }
    }
    // If depth wasn't sufficient, we can still use 'bestMoveFromTT' for ordering below
  }

  int playerWhoJustMoved = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  int evalScore = Evaluation::evaluatePositionHard(board, playerWhoJustMoved, lastX, lastY);
  if (lastX != -1 && evalScore >= MINIMAX_TERMINATION) {
    // Optional: Store this terminal state in TT? Could use a large depth.
    // transTable[currentHash] = TTEntry(evalScore, MAX_DEPTH + depth, std::make_pair(-1,-1),
    // EXACT);
    if (board->getCapturedStones().size() > 0) {
      board->applyCapture(true);
    }
    return evalScore;
  }

  if (depth == 0) {
    int quiescenceSearchScore = quiescenceSearch(board, alpha, beta, isMaximizing, lastX, lastY);
    if (board->getCapturedStones().size() > 0) {
      board->applyCapture(true);
    }
    return quiescenceSearchScore;
  }

  if (board->getCapturedStones().size() > 0) {
    board->applyCapture(true);
  }

  currentHash = board->getHash();  // *** Requires Board::getHash() ***
  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    int final_eval = Evaluation::evaluatePositionHard(board, currentPlayer, lastX, lastY);
    // Store this terminal evaluation in TT
    transTable[currentHash] = TTEntry(final_eval, depth, std::make_pair(-1, -1), EXACT);
    return final_eval;
  }

  int bestEval;  // Will hold the best score found for this node
  std::pair<int, int> bestMoveForNode =
      std::make_pair(-1, -1);  // Track best move found at this node
  bool processed_hash_move = false;

  // 1. Try the Hash Move first (if it exists and is valid)
  if (bestMoveFromTT.first != -1) {
    // Optional: Verify bestMoveFromTT is in 'moves'. Assume it is for now if hashing is correct.
    const std::pair<int, int> &move = bestMoveFromTT;
    UndoInfo info = board->makeMove(move.first, move.second);
    int nextPlayer = board->getNextPlayer();
    int eval =
        minimax(board, depth - 1, alpha, beta, nextPlayer, move.first, move.second, !isMaximizing);
    board->undoMove(info);
    processed_hash_move = true;  // Mark hash move as processed

    // Initialize bestEval and bestMoveForNode with the hash move's result
    bestEval = eval;
    bestMoveForNode = move;  // Initially assume hash move is best

    // Update alpha/beta based on hash move result
    if (isMaximizing) {
      alpha = std::max(alpha, eval);
    } else {  // Minimizing
      beta = std::min(beta, eval);
    }

    // Check for cutoff immediately after the hash move
    if (alpha >= beta) {
      // Hash move caused cutoff. Store TT entry.
      BoundType flag = isMaximizing ? LOWERBOUND : UPPERBOUND;  // Cutoff means we found a bound
      transTable[currentHash] = TTEntry(bestEval, depth, bestMoveForNode, flag);
      return bestEval;  // Return score causing cutoff
    }
  } else {
    // Initialize bestEval if hash move wasn't processed
    bestEval = isMaximizing ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
  }

  // ***** MOVE ORDERING (C++98 VERSION) *****

  // 1. Create vector for scored moves
  std::vector<ScoredMove> scored_moves;
  scored_moves.reserve(moves.size());  // Good practice, C++98 compatible

  // 2. Calculate score ONCE for each move (using iterators)
  for (std::vector<std::pair<int, int> >::const_iterator it = moves.begin(); it != moves.end();
       ++it) {
    const std::pair<int, int> &m = *it;  // Explicit type for the move

    if (processed_hash_move && m == bestMoveFromTT) {
      continue;
    }

    int move_score = Evaluation::evaluatePositionHard(board, currentPlayer, m.first, m.second);

    // RANDOM TERMINATION PRUNINIG
    // if (move_score >= MINIMAX_TERMINATION) {
    //   return move_score;
    // }
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

  // 4. Iterate through the *sorted* scored_moves (using iterators)
  for (std::vector<ScoredMove>::const_iterator it = scored_moves.begin(); it != scored_moves.end();
       ++it) {
    const ScoredMove &scored_move = *it;
    const std::pair<int, int> &move = scored_move.move;

    UndoInfo info = board->makeMove(move.first, move.second);
    int nextPlayer = board->getNextPlayer();
    // --- Recursive Call ---
    int eval =
        minimax(board, depth - 1, alpha, beta, nextPlayer, move.first, move.second, !isMaximizing);
    board->undoMove(info);
    // --- Evaluation & Pruning ---
    if (isMaximizing) {
      if (eval > bestEval) {  // Found a new best move
        bestEval = eval;
        bestMoveForNode = move;             // Update best move for TT
        alpha = std::max(alpha, bestEval);  // Update alpha
      }
      if (alpha >= beta) {  // Beta cutoff
        // Store killer move
        if (killerMoves[depth][0] != move && killerMoves[depth][1] != move) {
          killerMoves[depth][1] = killerMoves[depth][0];
          killerMoves[depth][0] = move;
        }
        // Store TT entry causing cutoff
        transTable[currentHash] =
            TTEntry(bestEval, depth, bestMoveForNode, LOWERBOUND);  // Fail high => Lower bound
        return bestEval;
      }
    } else {                  // Minimizing
      if (eval < bestEval) {  // Found a new best move
        bestEval = eval;
        bestMoveForNode = move;           // Update best move for TT
        beta = std::min(beta, bestEval);  // Update beta
      }
      if (alpha >= beta) {  // Alpha cutoff
        // Store killer move
        if (killerMoves[depth][0] != move && killerMoves[depth][1] != move) {
          killerMoves[depth][1] = killerMoves[depth][0];
          killerMoves[depth][0] = move;
        }
        // Store TT entry causing cutoff
        transTable[currentHash] =
            TTEntry(bestEval, depth, bestMoveForNode, UPPERBOUND);  // Fail low => Upper bound
        return bestEval;
      }
    }
  }  // End loop over sorted moves
     // --- Store Final Result in TT Before Returning ---
  // If we reach here, no cutoff occurred for the remaining moves.
  // Determine the correct flag based on the initial alpha.
  BoundType flag;
  if (bestEval <= initial_alpha) {
    flag = UPPERBOUND;  // Failed low relative to initial window
  } else {
    // Since beta cutoff didn't happen, bestEval must be < beta.
    // And since we are here, bestEval > initial_alpha.
    flag = EXACT;  // Score is within the initial alpha-beta window
  }

  // Store the final result. Consider TT replacement strategy (e.g., only store if depth is >=
  // existing depth) if TT gets full. Simple overwrite for now:
  transTable[currentHash] = TTEntry(bestEval, depth, bestMoveForNode, flag);

  return bestEval;
}

std::pair<int, int> getBestMove(Board *board, int depth) {
  int bestScore = std::numeric_limits<int>::min();
  std::pair<int, int> bestMove = std::make_pair(-1, -1);

  int currentPlayer = board->getNextPlayer();
  int root_alpha = std::numeric_limits<int>::min();  // Initial alpha = -infinity
  int root_beta = std::numeric_limits<int>::max();   // Initial beta = +infinity

  uint64_t initialHash = board->getHash();                      // Get hash of the root position
  std::pair<int, int> bestMoveFromTT = std::make_pair(-1, -1);  // Initialize hash move candidate
  boost::unordered_map<uint64_t, TTEntry>::iterator tt_it = transTable.find(initialHash);

  if (tt_it != transTable.end()) {
    const TTEntry &entry = tt_it->second;
    // We primarily care about the best move for ordering at the root.
    // We could potentially use the score if depth is sufficient and flag is EXACT,
    // but the main search loop below will confirm it anyway.
    bestMoveFromTT = entry.bestMove;
    std::cout << "TT Hit at root. Suggests move: (" << bestMoveFromTT.first << ","
              << bestMoveFromTT.second << ")" << std::endl;
  }

  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) return bestMove;

  initKillerMoves();

  bool processed_hash_move = false;

  // --- Process Hash Move First (if valid) ---
  if (bestMoveFromTT.first != -1) {
    // Check if the TT move is actually in the list of legal moves
    bool found = false;
    for (size_t i = 0; i < moves.size(); ++i) {
      if (moves[i] == bestMoveFromTT) {
        found = true;
        break;
      }
    }

    if (found) {
      std::cout << "Processing Hash Move first: (" << bestMoveFromTT.first << ","
                << bestMoveFromTT.second << ")" << std::endl;
      const std::pair<int, int> &move = bestMoveFromTT;
      int playerMakingMove = board->getNextPlayer();  // Should be currentPlayer

      UndoInfo info = board->makeMove(move.first, move.second);
      // Call minimax for the opponent (isMaximizing = false)
      int score = minimax(board, depth - 1, root_alpha, root_beta, playerMakingMove, move.first,
                          move.second, false);
      board->undoMove(info);

      // Update best score and move found so far
      bestScore = score;
      bestMove = move;
      root_alpha = std::max(root_alpha, score);  // Update alpha
      processed_hash_move = true;
      std::cout << "Hash Move score: " << score << std::endl;

      // Note: In a full root implementation with pruning, we might check
      // if root_alpha >= root_beta here, but this function evaluates all moves.
    } else {
      std::cout << "Hash Move from TT was not found in legal moves." << std::endl;
      bestMoveFromTT = std::make_pair(-1, -1);  // Invalidate TT move if not legal
    }
  }

  std::vector<ScoredMove> scored_moves;
  scored_moves.reserve(moves.size());  // Good practice, C++98 compatible

  // 2. Calculate score ONCE for each move (using iterators)
  for (std::vector<std::pair<int, int> >::const_iterator it = moves.begin(); it != moves.end();
       ++it) {
    const std::pair<int, int> &m = *it;  // Explicit type for the move

    if (processed_hash_move && m == bestMoveFromTT) {
      continue;
    }

    int move_score = Evaluation::evaluatePositionHard(board, currentPlayer, m.first, m.second);
    // Check killer status (ensure 'm' can be compared to killerMoves elements)
    bool is_killer = false;
    if (depth >= 0 /* && depth < MAX_DEPTH_LIMIT */) {  // Add bounds check if necessary
      is_killer = (m == killerMoves[depth][0] || m == killerMoves[depth][1]);
    }
    // Add to vector using push_back and explicit constructor call
    scored_moves.push_back(ScoredMove(move_score, m, is_killer));
  }

  std::sort(scored_moves.begin(), scored_moves.end(), CompareScoredMovesMax());
  // ***** END PRE-CALCULATION & SORTING *****

  std::cout << "depth: " << depth << " player" << currentPlayer << std::endl;
  // Evaluate each candidate move.
  for (size_t i = 0; i < scored_moves.size(); ++i) {
    if (scored_moves[i].score > MINIMAX_TERMINATION) return scored_moves[i].move;
    int playerMakingMove = board->getNextPlayer();  // Get player before making move

    UndoInfo info = board->makeMove(scored_moves[i].move.first, scored_moves[i].move.second);
    int score = minimax(board, depth - 1, root_alpha, root_beta, playerMakingMove,
                        scored_moves[i].move.first, scored_moves[i].move.second, false);
    board->undoMove(info);
    if (score > bestScore) {
      bestScore = score;
      bestMove = scored_moves[i].move;
      root_alpha = std::max(root_alpha, score);
    }
  }

  if (bestMove.first != -1) transTable[initialHash] = TTEntry(bestScore, depth, bestMove, EXACT);
  std::cout << "final: " << board->getNextPlayer() << std::endl;
  return bestMove;
}

struct SearchResult {
  std::pair<int, int> bestMove;
  int score;
  int depthSearched;  // Depth actually completed

  SearchResult() : bestMove(-1, -1), score(std::numeric_limits<int>::min()), depthSearched(0) {}
};

// The main iterative deepening function
std::pair<int, int> iterativeDeepening(Board *board, int maxDepth, double timeLimitSeconds) {
  // Timekeeping (C++98 style using clock())
  clock_t startTime = clock();
  clock_t timeLimitClocks = (clock_t)(timeLimitSeconds * CLOCKS_PER_SEC);

  SearchResult bestResultSoFar;  // Store best result from completed depths
  int currentPlayer = board->getNextPlayer();

  initKillerMoves();  // Initialize killers at the start of ID

  // --- Iterative Deepening Loop ---
  for (int currentDepth = 1; currentDepth <= maxDepth; ++currentDepth) {
    std::cout << "Starting search at depth: " << currentDepth << std::endl;

    int bestScoreThisIteration = std::numeric_limits<int>::min();
    std::pair<int, int> bestMoveThisIteration = std::make_pair(-1, -1);
    int root_alpha = std::numeric_limits<int>::min();
    int root_beta = std::numeric_limits<int>::max();

    // --- Root Move Generation & Ordering ---
    uint64_t initialHash = board->getHash();
    std::pair<int, int> bestMoveFromPrevIterOrTT =
        bestResultSoFar.bestMove;  // Use prev iter's move first

    // Check TT for the root node for potentially better move suggestion or score estimate
    boost::unordered_map<uint64_t, TTEntry>::iterator tt_it = transTable.find(initialHash);
    if (tt_it != transTable.end()) {
      const TTEntry &entry = tt_it->second;
      // If TT suggests a move and we don't have one from prev iter, use TT's suggestion
      if (bestMoveFromPrevIterOrTT.first == -1 && entry.bestMove.first != -1) {
        bestMoveFromPrevIterOrTT = entry.bestMove;
        std::cout << "Using TT suggested move for ordering: (" << bestMoveFromPrevIterOrTT.first
                  << "," << bestMoveFromPrevIterOrTT.second << ")" << std::endl;
      }
      // Could potentially use entry.score as a guess for MTD(f) later or to center aspiration
      // window
    }

    std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
    if (moves.empty()) {
      std::cout << "No moves available." << std::endl;
      // Return immediately if no moves at depth 1, else return previous best
      return (currentDepth == 1) ? std::make_pair(-1, -1) : bestResultSoFar.bestMove;
    }

    // Prepare for sorting - prioritize the best move from previous iteration or TT
    std::vector<ScoredMove> scored_moves;
    scored_moves.reserve(moves.size());
    bool processed_priority_move = false;

    // Process the priority move first IF it's valid
    if (bestMoveFromPrevIterOrTT.first != -1) {
      bool found = false;
      for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i] == bestMoveFromPrevIterOrTT) {
          found = true;
          break;
        }
      }

      if (found) {
        // Evaluate the priority move first
        std::cout << "Processing Priority Move first: (" << bestMoveFromPrevIterOrTT.first << ","
                  << bestMoveFromPrevIterOrTT.second << ")" << std::endl;
        const std::pair<int, int> &move = bestMoveFromPrevIterOrTT;
        int playerMakingMove = board->getNextPlayer();
        UndoInfo info = board->makeMove(move.first, move.second);
        int score = minimax(board, currentDepth - 1, root_alpha, root_beta, playerMakingMove,
                            move.first, move.second, false);
        board->undoMove(info);
        processed_priority_move = true;

        bestScoreThisIteration = score;
        bestMoveThisIteration = move;
        root_alpha = std::max(root_alpha, score);
        std::cout << "Priority Move score: " << score << std::endl;
      } else {
        std::cout << "Priority Move not found in legal moves." << std::endl;
        bestMoveFromPrevIterOrTT = std::make_pair(-1, -1);  // Invalidate
      }
    }

    // Score and prepare remaining moves for sorting
    for (std::vector<std::pair<int, int> >::const_iterator it = moves.begin(); it != moves.end();
         ++it) {
      const std::pair<int, int> &m = *it;
      if (processed_priority_move && m == bestMoveFromPrevIterOrTT) {
        continue;
      }

      int move_score_heuristic =
          Evaluation::evaluatePositionHard(board, currentPlayer, m.first, m.second);
      if (move_score_heuristic >= MINIMAX_TERMINATION) {
        std::cout << "Immediate heuristic win found at root: (" << m.first << "," << m.second
                  << ") during depth " << currentDepth << std::endl;
        // Store this iteration's result before returning
        bestResultSoFar.bestMove = m;
        bestResultSoFar.score = move_score_heuristic;
        bestResultSoFar.depthSearched = currentDepth;
        // Optional TT Store
        // transTable[initialHash] = TTEntry(move_score_heuristic, currentDepth, m, EXACT);
        return m;  // Return winning move immediately
      }
      bool is_killer = false;   // Killers might need careful indexing with ID (e.g., relative depth
                                // or flat array) Let's use currentDepth for now, assuming MAX_DEPTH
                                // is large enough in killer array
      if (currentDepth >= 0) {  // Index check
        is_killer = (m == killerMoves[currentDepth][0] || m == killerMoves[currentDepth][1]);
      }
      scored_moves.push_back(ScoredMove(move_score_heuristic, m, is_killer));
    }

    // Sort remaining moves
    std::sort(scored_moves.begin(), scored_moves.end(), CompareScoredMovesMax());

    // --- Search Remaining Moves ---
    for (size_t i = 0; i < scored_moves.size(); ++i) {
      // *** Check Time Limit ***
      clock_t currentTime = clock();
      if ((currentTime - startTime) >= timeLimitClocks) {
        std::cout << "Time limit exceeded during depth " << currentDepth << "!" << std::endl;
        // Return best move from PREVIOUS completed iteration
        return bestResultSoFar.bestMove;
      }

      const std::pair<int, int> &move = scored_moves[i].move;
      int playerMakingMove = board->getNextPlayer();
      UndoInfo info = board->makeMove(move.first, move.second);
      int score = minimax(board, currentDepth - 1, root_alpha, root_beta, playerMakingMove,
                          move.first, move.second, false);
      board->undoMove(info);

      std::cout << "  Depth " << currentDepth << " Move (" << move.first << "," << move.second
                << ") score: " << score << std::endl;

      if (score > bestScoreThisIteration) {
        bestScoreThisIteration = score;
        bestMoveThisIteration = move;
        root_alpha = std::max(root_alpha, score);
        std::cout << "  New best score for depth " << currentDepth << ": " << bestScoreThisIteration
                  << " for move (" << bestMoveThisIteration.first << ","
                  << bestMoveThisIteration.second << ")" << std::endl;
      }
    }  // End loop through remaining moves

    // --- Iteration Complete ---

    // Check time *after* completing the depth's search loop
    clock_t currentTime = clock();
    if ((currentTime - startTime) >= timeLimitClocks && currentDepth > 1) {
      std::cout << "Time limit exceeded AFTER completing depth " << currentDepth << " loop."
                << std::endl;
      // We finished the iteration just now, but might not have time for the next one.
      // Still update bestResultSoFar with the results of the depth we just finished.
      // The return below will handle returning the result from *this* depth.
    }

    // Check if a valid move was found in this iteration
    if (bestMoveThisIteration.first != -1) {
      // Store result of the completed iteration
      bestResultSoFar.bestMove = bestMoveThisIteration;
      bestResultSoFar.score = bestScoreThisIteration;
      bestResultSoFar.depthSearched = currentDepth;

      // Store in TT
      std::cout << "Storing result in TT for depth " << currentDepth
                << ": Score=" << bestScoreThisIteration << ", Move=(" << bestMoveThisIteration.first
                << "," << bestMoveThisIteration.second << ")" << std::endl;
      transTable[initialHash] =
          TTEntry(bestScoreThisIteration, currentDepth, bestMoveThisIteration, EXACT);

      std::cout << "Finished depth " << currentDepth << ". Best move: ("
                << bestMoveThisIteration.first << "," << bestMoveThisIteration.second
                << ") Score: " << bestScoreThisIteration << std::endl;

    } else if (!processed_priority_move) {
      // This case should ideally not happen if generateCandidateMoves returned >0 moves initially
      // Means no move yielded a score better than -infinity. Could be error or draw state?
      std::cout << "Warning: No best move found for depth " << currentDepth
                << " although moves exist." << std::endl;
      // If no move improved score at depth 1, return default or first legal move?
      // Stick with returning previous best for safety if depth > 1.
      if (currentDepth == 1) {
        // Maybe return moves[0]? Or stick with default (-1,-1).
        return std::make_pair(-1, -1);
      } else {
        return bestResultSoFar.bestMove;
      }
    }
    // If only processed hash move, bestResultSoFar still got updated earlier

    // Optional: Early exit if score indicates forced mate?
    // if (abs(bestResultSoFar.score) >= (WIN_SCORE - maxDepth)) // Adjust score based on depth for
    // mate distance
    // {
    //    std::cout << "Forced mate found at depth " << currentDepth << std::endl;
    //    break; // Exit ID loop
    // }

    // Check time *before* starting next iteration (redundant with check inside loop, but safe)
    if ((clock() - startTime) >= timeLimitClocks) {
      std::cout << "Time limit reached after completing depth " << currentDepth << "." << std::endl;
      break;  // Exit ID loop
    }

  }  // End Iterative Deepening Loop

  std::cout << "Iterative deepening finished. Returning best move from depth "
            << bestResultSoFar.depthSearched << std::endl;
  return bestResultSoFar.bestMove;  // Return best move from the deepest fully completed search
}

}  // namespace Minimax
