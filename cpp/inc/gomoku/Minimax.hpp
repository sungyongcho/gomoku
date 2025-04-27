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

static boost::unordered_map<uint64_t, TTEntry> transTable;

namespace Minimax {

void initKillerMoves();
std::vector<std::pair<int, int> > generateCandidateMoves(Board *&board);

void printBoardWithCandidates(Board *&board, const std::vector<std::pair<int, int> > &candidates);

std::pair<int, int> getBestMove(Board *board, int depth);
std::pair<int, int> iterativeDeepening(Board *board, int maxDepth, double timeLimitSeconds);

}  // namespace Minimax

#endif  // MINMAX_HPP
