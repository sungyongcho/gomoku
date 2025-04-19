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

// Window extraction settings.
// SIDE_WINDOW_SIZE: the number of cells to extract on each side (excluding center).
#define SIDE_WINDOW_SIZE_TMP 4
// Combined window size always equals 2*SIDE_WINDOW_SIZE + 1 (center cell + cells on both sides).
#define COMBINED_WINDOW_SIZE_TMP (2 * SIDE_WINDOW_SIZE_TMP + 1)
// - Shifting 1 left by (2 * COMBINED_WINDOW_SIZE) is equivalent to 2^(2 * COMBINED_WINDOW_SIZE),
//   which is the total number of unique patterns that can be represented.
#define LOOKUP_TABLE_SIZE_TMP (1 << (2 * COMBINED_WINDOW_SIZE_TMP))

#define MAX_DEPTH 5
// Bound types used for alpha-beta entries.
enum BoundType { EXACT, LOWERBOUND, UPPERBOUND };

// Structure for transposition table entries.
struct TTEntry {
  int score;                     // Evaluation score
  int depth;                     // Depth at which the evaluation was computed
  std::pair<int, int> bestMove;  // Best move from this state (if available)
  BoundType flag;                // Flag indicating whether score is EXACT, a lower, or upper bound.
};

// Global transposition table (using std::map for simplicity in C++98).
static std::map<uint64_t, TTEntry> transTable;

// killerMoves[depth][0] is the primary killer, [1] is the secondary.
// Initialize all killer moves to (-1, -1) to indicate "no move".
namespace Minimax {

void initKillerMoves();
std::vector<std::pair<int, int> > generateCandidateMoves(Board *&board);

void printBoardWithCandidates(Board *&board, const std::vector<std::pair<int, int> > &candidates);

std::pair<int, int> getBestMove(Board *board, int depth);
std::pair<int, int> iterativeDeepening(Board *board, int maxDepth, double timeLimitSeconds);

void simulateAIBattle(Board *pBoard, int searchDepth, int numTurns);

}  // namespace Minimax

#endif  // MINMAX_HPP
