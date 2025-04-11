#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <algorithm>
#include <cmath>

#include "Board.hpp"
#include "Gomoku.hpp"

#define GOMOKU 10000000
#define CAPTURE_WIN 11000000
#define FORCED_WIN_SEQUENCE 950000
#define DOUBLE_OPEN_FOUR 950000
#define OPEN_FOUR 700000
#define CLOSED_FOUR 400000
#define DOUBLE_THREAT 800000
#define FORK 750000
#define CAPTURE_LEADING 500000
#define OPEN_THREE 200000
#define CLOSED_THREE 100000
#define DEFENSIVE_BLOCK 350000
#define COUNTER_THREAT 300000
#define CONTINUOUS_LINE_4 100000
#define CONTINUOUS_LINE_3 1000
#define CONTINUOUS_LINE_2 100
#define CONTINUOUS_LINE_1 10
#define BLOCK_LINE_5 999000
#define BLOCK_LINE_4 10000
#define BLOCK_LINE_3 100
#define BLOCK_LINE_2 10
#define BLOCK_LINE_1 1
#define CAPTURE_SCORE 200000
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

struct PatternCounts {
  int openFourCount;        // Used for forced win / double open four detection.
  int closedFourCount;      // Used for forced win / double open four detection.
  int openThreeCount;       // Used for forced win / double open four detection.
  int closedThreeCount;     // Used for forced win / double open four detection.
  int threatCount;          // Counts open/closed three threats.
  int captureCount;         // Counts capture opportunities ("OXXO").
  int defensiveBlockCount;  // Counts when this move blocks an opponent pattern.
  int captureBlockCount;

  PatternCounts()
      : openFourCount(0),
        threatCount(0),
        captureCount(0),
        defensiveBlockCount(0),
        captureBlockCount(0) {}
};

struct EvaluationEntry {
  int score;
  PatternCounts counts;

  EvaluationEntry() : score(0), counts() {}

  EvaluationEntry(int s, const PatternCounts &pc) : score(s), counts(pc) {}
  // Overload operator += to combine two EvaluationEntries.
  EvaluationEntry &operator+=(const EvaluationEntry &other) {
    score += other.score;
    counts.openFourCount += other.counts.openFourCount;
    counts.threatCount += other.counts.threatCount;
    counts.captureCount += other.counts.captureCount;
    counts.defensiveBlockCount += other.counts.defensiveBlockCount;
    counts.captureBlockCount += other.counts.captureBlockCount;
    return *this;
  }
};

const EvaluationEntry INVALID_ENTRY(INVALID_PATTERN, PatternCounts());

extern int patternScoreTablePlayerOne[LOOKUP_TABLE_SIZE];
extern int patternScoreTablePlayerTwo[LOOKUP_TABLE_SIZE];

static EvaluationEntry patternPlayerOne[LOOKUP_TABLE_SIZE];
static EvaluationEntry patternPlayerTwo[LOOKUP_TABLE_SIZE];

static const int continuousScores[6] = {
    0, CONTINUOUS_LINE_1, CONTINUOUS_LINE_2, CONTINUOUS_LINE_3, CONTINUOUS_LINE_4, GOMOKU};
static const int blockScores[6] = {
    0, BLOCK_LINE_1, BLOCK_LINE_2, BLOCK_LINE_3, BLOCK_LINE_4, BLOCK_LINE_5};

bool isValidBackwardPattern(unsigned int sidePattern);
bool isValidForwardPattern(unsigned int sidePattern);

bool isCaptureWarning(int side, int player, bool reverse);
bool isCaptureVulnerable(int forward, int backward, int player);

void slideWindowContinuous(int side, int player, bool reverse, int &continuous, bool &isClosedEnd,
                           int &continuousEmpty);
void slideWindowBlock(int side, int player, bool reverse, int &blockContinuous, bool &isClosedEnd);

void initCombinedPatternScoreTables();
void initCombinedPatternScoreTablesHard();

int checkVPattern(Board *board, int player, int x, int y, int i);
int checkCapture(unsigned int side, unsigned int player);

unsigned int reversePattern(unsigned int pattern, int windowSize);

int evaluatePosition(Board *&board, int player, int x, int y);

int evaluatePositionHard(Board *&board, int player, int x, int y);

int getEvaluationRating(int score);

}  // namespace Evaluation
#endif  // EVALUATION_HPP
