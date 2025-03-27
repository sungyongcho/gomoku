#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <algorithm>
#include <cmath>

#include "Board.hpp"
#include "Gomoku.hpp"

#define GOMOKU 10000000
#define CONTINUOUS_LINE_4 100000
#define CONTINUOUS_LINE_3 1000
#define CONTINUOUS_LINE_2 100
#define CONTINUOUS_LINE_1 10
#define BLOCK_LINE_5 999000
#define BLOCK_LINE_4 10000
#define BLOCK_LINE_3 100
#define BLOCK_LINE_2 10
#define BLOCK_LINE_1 1
#define OPEN_SINGLE_STONE 10
#define CAPTURE_SCORE 1000
#define WINDOW_CENTER_VALUE 0
// TODO needs to check
#define INVALID_PATTERN -1337

// Window extraction settings.
// SIDE_WINDOW_SIZE: the number of cells to extract on each side (excluding center).
#define SIDE_WINDOW_SIZE 4
// Combined window size always equals 2*SIDE_WINDOW_SIZE + 1 (center cell + cells on both sides).
#define COMBINED_WINDOW_SIZE (2 * SIDE_WINDOW_SIZE + 1)
// - Shifting 1 left by (2 * COMBINED_WINDOW_SIZE) is equivalent to 2^(2 * COMBINED_WINDOW_SIZE),
//   which is the total number of unique patterns that can be represented.
#define LOOKUP_TABLE_SIZE (1 << (2 * COMBINED_WINDOW_SIZE))

namespace Evaluation {
// Global lookup table for combined patterns.
extern int patternScoreTablePlayerOne[LOOKUP_TABLE_SIZE];
extern int patternScoreTablePlayerTwo[LOOKUP_TABLE_SIZE];

static const int continuousScores[5] = {0, CONTINUOUS_LINE_2, CONTINUOUS_LINE_3, CONTINUOUS_LINE_4,
                                        GOMOKU};
static const int blockScores[5] = {0, BLOCK_LINE_2, BLOCK_LINE_3, BLOCK_LINE_4, GOMOKU};

// Initializes the lookup table for combined patterns.
void initCombinedPatternScoreTables();

// // Evaluates the board position at (x,y) for the given player.
// // Returns a heuristic score computed by summing the scores from the four axes.
int evaluatePosition(Board *&board, int player, int x, int y);

// // Helper: Evaluates one axis (direction dx,dy) at (x,y) by combining the forward
// // and backward windows with the center move, then using the lookup table.
// int evaluateCombinedAxis(Board *board, int player, int x, int y, int dx, int dy);

// // Helper: Extracts a window of 'length' cells as a bit-packed integer,
// // starting from (x,y) moving in direction (dx,dy). Out-of-bound cells are marked with
// // OUT_OF_BOUNDS.
// unsigned int extractLineAsBits(Board *board, int x, int y, int dx, int dy, int length);

// // Helper: Reverses a bit-packed pattern over 'windowSize' cells (each cell: 2 bits).
// unsigned int reversePattern(unsigned int pattern, int windowSize);

// // Helper: Given a combined pattern (of COMBINED_WINDOW_SIZE cells),
// // decode it into an array and return a heuristic score based on the contiguous run
// // (including the center), open ends, and capture opportunities.
// int evaluateCombinedPattern(int combinedPattern, int player);

}  // namespace Evaluation
#endif  // EVALUATION_HPP
