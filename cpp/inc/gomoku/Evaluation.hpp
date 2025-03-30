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
#define CAPTURE_SCORE 100
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

extern int patternScoreTablePlayerOne[LOOKUP_TABLE_SIZE];
extern int patternScoreTablePlayerTwo[LOOKUP_TABLE_SIZE];

static const int continuousScores[6] = {
    0, CONTINUOUS_LINE_1, CONTINUOUS_LINE_2, CONTINUOUS_LINE_3, CONTINUOUS_LINE_4, GOMOKU};
static const int blockScores[6] = {
    0, BLOCK_LINE_1, BLOCK_LINE_2, BLOCK_LINE_3, BLOCK_LINE_4, BLOCK_LINE_5};

void initCombinedPatternScoreTables();

int evaluatePosition(Board *&board, int player, int x, int y);

int getEvaluationRating(int score);

}  // namespace Evaluation
#endif  // EVALUATION_HPP
