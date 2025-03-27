#ifndef MINMAX_HPP
#define MINMAX_HPP

#include <algorithm>
#include <cmath>
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

namespace Minimax {

std::vector<std::pair<int, int> > generateCandidateMoves(Board *&board);

void printBoardWithCandidates(Board *&board, const std::vector<std::pair<int, int> > &candidates);

std::pair<int, int> getBestMove(Board *board, int depth);

void simulateAIBattle(Board *pBoard, int searchDepth, int numTurns);

}  // namespace Minimax

#endif  // MINMAX_HPP
