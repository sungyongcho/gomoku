#ifndef MINMAX_HPP
#define MINMAX_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Board.hpp"
#include "Gomoku.hpp"
#include "Rules.hpp"

#define MAX_DEPTH 3
// Bound types used for alpha-beta entries.
enum BoundType { EXACT, LOWERBOUND, UPPERBOUND };

// Structure for transposition table entries.
struct TTEntry {
  int score;                     // Evaluation score
  int depth;                     // Depth at which the evaluation was computed
  std::pair<int, int> bestMove;  // Best move from this state (if available)
  BoundType flag;                // Flag indicating whether score is EXACT, a lower, or upper bound.
};

static std::map<uint64_t, TTEntry> transTable;

namespace Minimax {

void initKillerMoves();
std::vector<std::pair<int, int> > generateCandidateMoves(Board *&board);

void printBoardWithCandidates(Board *&board, const std::vector<std::pair<int, int> > &candidates);

std::pair<int, int> getBestMove(Board *board, int depth);
std::pair<int, int> iterativeDeepening(Board *board, int maxDepth, double timeLimitSeconds);

}  // namespace Minimax

#endif  // MINMAX_HPP
