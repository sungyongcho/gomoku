#include "Minimax.hpp"

#include <cstdlib>
#include <ctime>
#include <limits>
#include <sstream>

#include "Evaluation.hpp"

boost::unordered_map<uint64_t, TTEntry> transTable;

namespace Minimax {

// ---- Constants & shared state -------------------------------------------------

static const uint64_t rowMask = ((uint64_t)1 << BOARD_SIZE) - 1;
static std::pair<int, int> killerMoves[MAX_DEPTH + 1][2];
static const std::pair<int, int> kInvalidMove(-1, -1);

// ---- Bitboard helpers ---------------------------------------------------------

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

// ---- Move generation helpers --------------------------------------------------

inline uint64_t computeCandidateMask(uint64_t occupancyRow, uint64_t neighborRow) {
  return neighborRow & (~occupancyRow) & rowMask;
}

inline int initialExtreme(bool isMaximizing) {
  return isMaximizing ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
}

inline bool timeExceeded(clock_t start, clock_t limit) {
  return (clock() - start) >= limit;
}

inline void updateBestAndBounds(bool isMaximizing, int eval, const std::pair<int, int> &mv,
                                int &bestEval, std::pair<int, int> &bestMove, int &alpha,
                                int &beta) {
  if (isMaximizing) {
    if (eval > bestEval) {
      bestEval = eval;
      bestMove = mv;
    }
    alpha = std::max(alpha, eval);
  } else {
    if (eval < bestEval) {
      bestEval = eval;
      bestMove = mv;
    }
    beta = std::min(beta, eval);
  }
}

inline bool shouldIncludeMove(Board *board, int col, int row, int player,
                              bool enableDoubleThreeRestriction) {
  if (!enableDoubleThreeRestriction) return true;
  if (!Rules::detectDoublethree(*board, col, row, player)) return true;

  return false;
}

inline bool isCaptureMove(Board *board, int col, int row, int player) {
  return Rules::detectCaptureStonesNotStore(*board, col, row, player);
}

// ---- Move generation ----------------------------------------------------------

// Generate candidate moves using row-based neighbor mask.
std::vector<std::pair<int, int> > generateCandidateMoves(Board *&board) {
  std::vector<std::pair<int, int> > moves;
  uint64_t occupancy[BOARD_SIZE];
  uint64_t neighbor[BOARD_SIZE];
  int nextPlayer = board->getNextPlayer();
  bool enableDoubleThreeRestriction = board->getEnableDoubleThreeRestriction();

  board->getOccupancy(occupancy);
  computeNeighborMask(occupancy, neighbor);

  for (int row = 0; row < BOARD_SIZE; row++) {
    uint64_t candidates = computeCandidateMask(occupancy[row], neighbor[row]);
    for (int col = 0; col < BOARD_SIZE; col++) {
      if (candidates & (1ULL << col)) {
        if (shouldIncludeMove(board, col, row, nextPlayer, enableDoubleThreeRestriction))
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
  int nextPlayer = board->getNextPlayer();

  board->getOccupancy(occupancy);
  computeNeighborMask(occupancy, neighbor);

  for (int row = 0; row < BOARD_SIZE; row++) {
    uint64_t candidates = computeCandidateMask(occupancy[row], neighbor[row]);
    for (int col = 0; col < BOARD_SIZE; col++) {
      if (candidates & (1ULL << col)) {
        if (isCaptureMove(board, col, row, nextPlayer))
          moves.push_back(std::make_pair(col, row));
      }
    }
  }
  return moves;
}

// ---- Debug utilities ----------------------------------------------------------

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

// ---- Killer moves -------------------------------------------------------------

void initKillerMoves() {
  for (int d = 0; d < MAX_DEPTH; ++d) {
    killerMoves[d][0] = kInvalidMove;
    killerMoves[d][1] = kInvalidMove;
  }
}

// Helper: check if a move is a killer move for a given depth.
bool isKillerMove(int depth, const std::pair<int, int> &move) {
  return (killerMoves[depth][0] == move || killerMoves[depth][1] == move);
}

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

// ---- Search helpers -----------------------------------------------------------

int quiescenceSearch(Board *board, int alpha, int beta, bool isMaximizing, int x, int y, int depth,
                     EvalFn evalFn) {
  // 1. Evaluate Stand-Pat Score
  //    Perspective is crucial. Evaluate from the point of view of the player whose turn it is.
  int playerWhoseTurnItIs = board->getNextPlayer();
  // Use -1,-1 or appropriate dummy coords if last move isn't relevant here
  int stand_pat_score = (*evalFn)(board, playerWhoseTurnItIs, x, y);
  // 2. Initial Pruning based on Stand-Pat
  if (isMaximizing) {
    if (stand_pat_score >= beta) {
      board->flushCaptures();
      return stand_pat_score;  // Fail-soft beta cutoff
    }
    alpha = std::max(alpha, stand_pat_score);
  } else {  // Minimizing
    if (stand_pat_score <= alpha) {
      board->flushCaptures();
      return stand_pat_score;  // Fail-soft alpha cutoff
    }
    beta = std::min(beta, stand_pat_score);
  }

  board->flushCaptures();
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
                                captureMoves[i].second, depth + 1, evalFn);
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

inline bool probeTT(Board *board, int depth, int &alpha, int &beta, std::pair<int, int> &bestMove,
                    int &scoreOut) {
  uint64_t h = board->getHash();
  boost::unordered_map<uint64_t, TTEntry>::iterator it = transTable.find(h);
  if (it == transTable.end()) return false;

  const TTEntry &e = it->second;
  bestMove = e.bestMove;

  if (e.depth < depth) return false;  // only ordering info
  scoreOut = e.score;

  if (e.flag == EXACT) {
    board->flushCaptures();
    return true;
  }
  if (e.flag == LOWERBOUND)
    alpha = std::max(alpha, scoreOut);
  else /*UPPERBOUND*/
    beta = std::min(beta, scoreOut);
  if (alpha >= beta) {
    board->flushCaptures();
    return true;
  }
  return false;
}

inline void storeTT(uint64_t hash, int depth, const std::pair<int, int> &mv, int score, int alpha0,
                    int beta) {
  BoundType flag = EXACT;
  if (score <= alpha0)
    flag = UPPERBOUND;
  else if (score >= beta)
    flag = LOWERBOUND;

  transTable[hash] = TTEntry(score, depth, mv, flag);
}

inline void scoreAndSortMoves(Board *board, const std::vector<std::pair<int, int> > &in, int player,
                              int depth, bool maxSide, std::vector<ScoredMove> &out, EvalFn eval) {
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    const std::pair<int, int> &m = in[i];
    int s = (*eval)(board, player, m.first, m.second);
    bool k = (killerMoves[depth][0] == m) || (killerMoves[depth][1] == m);
    out.push_back(ScoredMove(s, m, k));
  }
  if (maxSide)
    std::sort(out.begin(), out.end(), CompareScoredMovesMax());
  else
    std::sort(out.begin(), out.end(), CompareScoredMovesMin());
}

inline bool processHashMove(Board *board, const std::pair<int, int> &mv, int depth, int &alpha,
                            int &beta, bool isMaximizing, std::pair<int, int> &bestMoveOut,
                            int &bestEvalOut, EvalFn evalFn) {
  if (mv.first < 0) return false;  // no move to try

  // make
  UndoInfo ui = board->makeMove(mv.first, mv.second);

  // recurse
  int score = minimax(board, depth - 1, alpha, beta, board->getNextPlayer(), mv.first, mv.second,
                      !isMaximizing, evalFn);

  // undo
  board->undoMove(ui);

  // update bests
  bestEvalOut = score;
  bestMoveOut = mv;
  if (isMaximizing)
    alpha = std::max(alpha, score);
  else
    beta = std::min(beta, score);

  return alpha >= beta;
}

inline bool tryMoveAndCutoff(Board *board, const std::pair<int, int> &mv, int depth, int &alpha,
                             int &beta, bool isMaximizing, int initialAlpha, uint64_t currentHash,
                             std::pair<int, int> &bestMoveForNode, int &bestEval, EvalFn evalFn) {
  // 1) make
  UndoInfo ui = board->makeMove(mv.first, mv.second);

  // 2) recurse
  int nextPlayer = board->getNextPlayer();
  int eval = minimax(board, depth - 1, alpha, beta, nextPlayer, mv.first, mv.second, !isMaximizing,
                     evalFn);

  // 3) undo
  board->undoMove(ui);

  // 4) update bestEval & α/β
  updateBestAndBounds(isMaximizing, eval, mv, bestEval, bestMoveForNode, alpha, beta);

  // 5) on cutoff, record killer & TT and tell caller to exit
  if (alpha >= beta) {
    // killer
    if (killerMoves[depth][0] != mv && killerMoves[depth][1] != mv) {
      killerMoves[depth][1] = killerMoves[depth][0];
      killerMoves[depth][0] = mv;
    }
    // transposition table
    storeTT(currentHash, depth, bestMoveForNode, bestEval, initialAlpha, beta);
    return true;
  }

  return false;
}

// ---- Main search --------------------------------------------------------------

int minimax(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY,
            bool isMaximizing, EvalFn evalFn) {
  // --- Alpha-Beta Preamble ---
  int initial_alpha = alpha;            // Store original alpha for TT storing logic later
  uint64_t preHash = board->getHash();  // *** Requires Board::getHash() ***
  std::pair<int, int> bestMoveFromTT = kInvalidMove;  // Candidate Hash Move from TT
  int ttScore;
  if (probeTT(board, depth, alpha, beta, bestMoveFromTT, ttScore)) return ttScore;

  int playerWhoJustMoved = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  int evalScore = (*evalFn)(board, playerWhoJustMoved, lastX, lastY);
  if (lastX != -1 && evalScore >= MINIMAX_TERMINATION) {
    board->flushCaptures();
    return evalScore;
  }

  if (depth == 0) {
    if (board->getEnableCapture())
      evalScore = quiescenceSearch(board, alpha, beta, isMaximizing, lastX, lastY, depth, evalFn);
    board->flushCaptures();
    return evalScore;
  }

  int bestEval = initialExtreme(isMaximizing);
  std::pair<int, int> bestFromNode(kInvalidMove);

  if (processHashMove(board, bestMoveFromTT, depth, alpha, beta, isMaximizing, bestFromNode,
                      bestEval, evalFn)) {
    storeTT(preHash, depth, bestFromNode, bestEval, initial_alpha, beta);
    return bestEval;
  }

  board->flushCaptures();
  uint64_t currentHash = board->getHash();  // *** Requires Board::getHash() ***
  // Generate candidate moves.
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    int final_eval = (*evalFn)(board, currentPlayer, lastX, lastY);
    // Store this terminal evaluation in TT
    transTable[currentHash] = TTEntry(final_eval, depth, kInvalidMove, EXACT);
    return final_eval;
  }

  std::pair<int, int> bestMoveForNode = kInvalidMove;

  std::vector<ScoredMove> scored_moves;
  scoreAndSortMoves(board, moves, currentPlayer, depth, isMaximizing, scored_moves, evalFn);

  for (std::vector<ScoredMove>::const_iterator it = scored_moves.begin(); it != scored_moves.end();
       ++it) {
    if (tryMoveAndCutoff(board, it->move, depth, alpha, beta, isMaximizing, initial_alpha,
                         currentHash, bestMoveForNode, bestEval, evalFn)) {
      return bestEval;
    }
  }
  storeTT(currentHash, depth, bestMoveForNode, bestEval, initial_alpha, beta);

  return bestEval;
}

bool rootSearch(Board *board, int depth, int &alpha, int &beta, bool isMaximizing,
                std::pair<int, int> &bestMoveOut, int &bestScoreOut, clock_t startTime,
                clock_t timeLimitClocks, bool &timedOut, EvalFn evalFn) {
  // 0) pre‐timer check
  if (timeExceeded(startTime, timeLimitClocks)) {
    timedOut = true;
    return true;
  }

  // 1) TT lookup
  uint64_t h0 = board->getHash();
  int alpha0 = alpha;
  std::pair<int, int> ttMv(kInvalidMove);
  boost::unordered_map<uint64_t, TTEntry>::iterator it = transTable.find(h0);
  if (it != transTable.end()) {
    ttMv = it->second.bestMove;
    std::cout << "Using TT suggested move for ordering: (" << ttMv.first << "," << ttMv.second
              << ")" << std::endl;
  }

  // 2) try TT‐move
  if (processHashMove(board, ttMv, depth, alpha, beta, isMaximizing, bestMoveOut, bestScoreOut,
                      evalFn)) {
    storeTT(h0, depth, bestMoveOut, bestScoreOut, alpha0, beta);
    return true;
  }

  // 3) timer check again
  if (timeExceeded(startTime, timeLimitClocks)) {
    timedOut = true;
    return true;
  }

  // 4) generate & sort
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {
    std::cout << "No moves available." << std::endl;
    return false;
  }

  std::vector<ScoredMove> scored;
  scoreAndSortMoves(board, moves, board->getNextPlayer(), depth, isMaximizing, scored, evalFn);

  // 5) immediate heuristic win?
  if (!scored.empty() && scored[0].score >= MINIMAX_TERMINATION) {
    std::cout << "Immediate heuristic win found at root: (" << scored[0].move.first << ","
              << scored[0].move.second << ") during depth " << depth << std::endl;
    bestMoveOut = scored[0].move;
    bestScoreOut = scored[0].score;
    return true;
  }

  // 6) main loop
  bestScoreOut = initialExtreme(isMaximizing);

  for (size_t i = 0; i < scored.size(); ++i) {
    // inside‐loop timer check
    if (timeExceeded(startTime, timeLimitClocks)) {
      std::cout << "Time limit exceeded during depth " << depth << "!" << std::endl;
      timedOut = true;
      return true;
    }

    const std::pair<int, int> &mv = scored[i].move;
    UndoInfo ui = board->makeMove(mv.first, mv.second);
    int next = board->getNextPlayer();
    int val =
        minimax(board, depth - 1, alpha, beta, next, mv.first, mv.second, !isMaximizing, evalFn);
    board->undoMove(ui);

    // std::cout << "  Depth " << depth << " Move (" << mv.first << "," << mv.second
    //           << ") score: " << val << std::endl;

    updateBestAndBounds(isMaximizing, val, mv, bestScoreOut, bestMoveOut, alpha, beta);

    if (timeExceeded(startTime, timeLimitClocks)) {
      std::cout << "Time limit exceeded AFTER completing depth " << depth << " loop." << std::endl;
    }
  }

  // 7) store final TT entry
  storeTT(h0, depth, bestMoveOut, bestScoreOut, alpha0, beta);
  return false;
}

std::pair<int, int> getBestMove(Board *board, int depth, EvalFn evalFn) {
  int bestScore = std::numeric_limits<int>::min();
  std::pair<int, int> bestMove = kInvalidMove;

  int alpha = std::numeric_limits<int>::min();  // Initial alpha = -infinity
  int beta = std::numeric_limits<int>::max();   // Initial beta = +infinity

  rootSearch(board, depth, alpha, beta,
             true,  // maximizing at root
             bestMove, bestScore, evalFn);

  return bestMove;
}

// The main iterative deepening function
std::pair<int, int> iterativeDeepening(Board *board, int maxDepth, double timeLimitSeconds,
                                       EvalFn evalFn) {
  // Timekeeping (C++98 style using clock())
  clock_t start = clock();
  clock_t limit = (clock_t)(timeLimitSeconds * CLOCKS_PER_SEC);

  SearchResult bestSoFar;  // Store best result from completed depths

  initKillerMoves();  // Initialize killers at the start of ID

  int root_alpha = std::numeric_limits<int>::min();
  int root_beta = std::numeric_limits<int>::max();

  for (int d = 1; d <= maxDepth; ++d) {
    int bestScore;
    std::pair<int, int> bestMove(kInvalidMove);
    bool timedOut = false;

    // 1) First search with the *current* window
    bool cutoff = rootSearch(board, d, root_alpha, root_beta,
                             true,  // maximizing
                             bestMove, bestScore, start, limit, timedOut, evalFn);

    if (timedOut) break;  // ran out of time

    // 2) If no cutoff *and* score is outside [α,β], re-search full window
    if (!cutoff && (bestScore <= root_alpha || bestScore >= root_beta)) {
      root_alpha = std::numeric_limits<int>::min();
      root_beta = std::numeric_limits<int>::max();

      cutoff = rootSearch(board, d, root_alpha, root_beta, true, bestMove, bestScore, start, limit,
                          timedOut, evalFn);
      if (timedOut) break;  // time’s up in the full‐window search
    }

    // 3) If we cut off early (by TT or heuristic‐win), return immediately
    if (cutoff) return bestMove;

    // 4) Otherwise, record the fully completed result
    bestSoFar.bestMove = bestMove;
    bestSoFar.score = bestScore;
    bestSoFar.depthSearched = d;
  }

  return bestSoFar.bestMove;  // Return best move from the deepest fully completed search
}

int pvs(Board *board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY,
        bool isMaximizing, EvalFn evalFn) {
  int alphaOrig = alpha;  // for TT flag
  uint64_t hash = board->getHash();
  std::pair<int, int> ttMove(kInvalidMove);
  int ttScore;

  // ---- 1.  TT probe ---------------------------------------
  if (probeTT(board, depth, alpha, beta, ttMove, ttScore)) return ttScore;

  // ---- 2.  Terminal / quiescence --------------------------
  int playerJustMoved = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
  int eval = (*evalFn)(board, playerJustMoved, lastX, lastY);
  if ((lastX != -1 && eval >= MINIMAX_TERMINATION) || depth == 0) {
    board->flushCaptures();
    return eval;
  }
  board->flushCaptures();

  // ---- 3.  Generate & order moves -------------------------
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) {  // stalemate – evaluate statically
    board->flushCaptures();
    return (*evalFn)(board, currentPlayer, lastX, lastY);
  }
  std::vector<ScoredMove> scored;
  scoreAndSortMoves(board, moves, currentPlayer, depth, isMaximizing, scored, evalFn);

  bool firstChild = true;
  std::pair<int, int> bestMove(kInvalidMove);
  int bestEval = initialExtreme(isMaximizing);

  for (size_t i = 0; i < scored.size(); ++i) {
    const std::pair<int, int> &mv = scored[i].move;

    // Killer-move bookkeeping handled in tryMoveAndCutoff, so just call it.
    UndoInfo ui = board->makeMove(mv.first, mv.second);
    int next = board->getNextPlayer();

    int score;
    if (firstChild) {
      // full window
      score = pvs(board, depth - 1, alpha, beta, next, mv.first, mv.second, !isMaximizing, evalFn);
      firstChild = false;
    } else {
      // null window (width-0: alpha = beta = alpha + 1)
      score = pvs(board, depth - 1, alpha + 1, alpha + 1, next, mv.first, mv.second, !isMaximizing,
                  evalFn);
      // if it produced something interesting, re-search
      if (score > alpha && score < beta) {
        score =
            pvs(board, depth - 1, alpha, beta, next, mv.first, mv.second, !isMaximizing, evalFn);
      }
    }

    board->undoMove(ui);

    // ------- α/β update + best-move tracking -------------
    updateBestAndBounds(isMaximizing, score, mv, bestEval, bestMove, alpha, beta);
    if (alpha >= beta) {  // cut-off
      // killer-move table
      if (!isKillerMove(depth, mv)) {
        killerMoves[depth][1] = killerMoves[depth][0];
        killerMoves[depth][0] = mv;
      }
      break;
    }
  }

  // ---- 5.  Store in TT & return ---------------------------
  storeTT(hash, depth, bestMove, bestEval, alphaOrig, beta);
  return bestEval;
}

// ------------------------------------------------------------
// One-shot “find best move” helper (no iterative deepening).
// ------------------------------------------------------------
std::pair<int, int> getBestMovePVS(Board *board, int depth,
                                   EvalFn evalFn = &Evaluation::evaluatePositionHard) {
  initKillerMoves();
  std::vector<std::pair<int, int> > moves = generateCandidateMoves(board);
  if (moves.empty()) return std::make_pair(-1, -1);

  std::vector<ScoredMove> ordered;
  scoreAndSortMoves(board, moves, board->getNextPlayer(), depth, /*maxSide=*/true, ordered, evalFn);
  if (ordered[0].score >= MINIMAX_TERMINATION) {
    return ordered[0].move;
  }

  int alpha = std::numeric_limits<int>::min();
  int beta = std::numeric_limits<int>::max();
  std::pair<int, int> bestMove(kInvalidMove);
  int bestScore = std::numeric_limits<int>::min();

  for (size_t i = 0; i < ordered.size(); ++i) {
    const std::pair<int, int> &mv = ordered[i].move;
    UndoInfo ui = board->makeMove(mv.first, mv.second);
    int next = board->getNextPlayer();
    int score = pvs(board, depth - 1, alpha, beta, next, mv.first, mv.second,
                    /*isMaximizing=*/false, evalFn);

    board->undoMove(ui);
    if (score >= MINIMAX_TERMINATION) {
      return mv;
    }

    if (score > bestScore) {
      bestScore = score;
      bestMove = mv;
    }
    if (score > alpha) alpha = score;
  }
  return bestMove;
}

}  // namespace Minimax
