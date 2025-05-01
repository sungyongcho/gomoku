#ifndef MINMAX_HPP
#define MINMAX_HPP

#include <algorithm>
#include <boost/unordered_map.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "Board.hpp"
#include "Gomoku.hpp"
#include "Rules.hpp"

#define MAX_DEPTH 10
// Bound types used for alpha-beta entries.
enum BoundType { EXACT, LOWERBOUND, UPPERBOUND };

// Structure for transposition table entries.
struct TTEntry {
  int score;                     // Evaluation score
  int depth;                     // Depth at which the evaluation was computed
  std::pair<int, int> bestMove;  // Best move from this state (if available)
  BoundType flag;                // Flag indicating whether score is EXACT, a lower, or upper bound.

  TTEntry() : score(0), depth(-1), bestMove(-1, -1), flag(EXACT) {}

  // Parameterized constructor for convenience
  TTEntry(int s, int d, std::pair<int, int> mv, BoundType f)
      : score(s), depth(d), bestMove(mv), flag(f) {}
};

struct ScoredMove {
  int score;
  std::pair<int, int> move;
  bool is_killer;

  // Constructor (C++98 style)
  ScoredMove(int s, std::pair<int, int> m, bool ik) : score(s), move(m), is_killer(ik) {}
};

struct SearchResult {
  std::pair<int, int> bestMove;
  int score;
  int depthSearched;  // Depth actually completed

  SearchResult() : bestMove(-1, -1), score(std::numeric_limits<int>::min()), depthSearched(0) {}
};

static boost::unordered_map<uint64_t, TTEntry> transTable;

namespace Minimax {

void initKillerMoves();
std::vector<std::pair<int, int> > generateCandidateMoves(Board*& board);

void printBoardWithCandidates(Board*& board, const std::vector<std::pair<int, int> >& candidates);

int minimax(Board* board, int depth, int alpha, int beta, int currentPlayer, int lastX, int lastY,
            bool isMaximizing);
std::pair<int, int> getBestMove(Board* board, int depth);
std::pair<int, int> getBestMovePVS(Board* board, int depth);
std::pair<int, int> iterativeDeepening(Board* board, int maxDepth, double timeLimitSeconds);

bool probeTT(Board* board, int depth, int& alpha, int& beta, std::pair<int, int>& bestMove,
             int& scoreOut);
void storeTT(uint64_t hash, int depth, const std::pair<int, int>& bestMove, int score, int alpha0,
             int beta);

void scoreAndSortMoves(Board* board, const std::vector<std::pair<int, int> >& in, int player,
                       int depth, bool maxSide, std::vector<ScoredMove>& out);

bool processHashMove(Board* board, const std::pair<int, int>& mv, int depth, int& alpha, int& beta,
                     bool isMaximizing, std::pair<int, int>& bestMoveOut, int& bestEvalOut);

// Timer-aware root search (used by iterativeDeepening).
bool rootSearch(Board* board, int depth, int& alpha, int& beta, bool isMaximizing,
                std::pair<int, int>& bestMoveOut, int& bestScoreOut, clock_t startTime,
                clock_t timeLimitClocks, bool& timedOut);

// Timer-free overload (used by getBestMove).
inline bool rootSearch(Board* board, int depth, int& alpha, int& beta, bool isMaximizing,
                       std::pair<int, int>& bestMoveOut, int& bestScoreOut) {
  bool dummyTimedOut = false;
  return rootSearch(board, depth, alpha, beta, isMaximizing, bestMoveOut, bestScoreOut,
                    /*startTime=*/0,
                    /*timeLimitClocks=*/std::numeric_limits<clock_t>::max(), dummyTimedOut);
}

}  // namespace Minimax

#endif  // MINMAX_HPP
