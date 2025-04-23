#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <algorithm>
#include <cmath>
#include <set>

#include "Board.hpp"
#include "Gomoku.hpp"

#define CAPTURE_WIN 50000
#define GOMOKU 49000
#define CAPTURE_CRITICAL 19000
#define CAPTURE_BLOCK_CRITICAL 30000
#define BLOCK_CRITICAL_LINE 39500  // block open 3, 4

#define CONTINUOUS_OPEN_4 9000
#define CONTINUOUS_CLOSED_4 5000
#define CONTINUOUS_OPEN_3 4500
#define CONTINUOUS_CLOSED_3 4000  // It blocks a capture
#define CONTINUOUS_OPEN_2 1000

#define BLOCK_GOMOKU 20000
#define BLOCK_OPEN_2 1800
#define BLOCK_OPEN_1 1500
#define PERFECT_CRITICAL_LINE 18000

#define CAPTURE 6000
#define CAPTURE_THREAT 3300
#define THREAT 2000
#define THREAT_BLOCK 3000
#define CENTER_BONUS 1000
#define WINDOW_CENTER_VALUE 0
// TODO needs to check
#define INVALID_PATTERN -133
#define CAPTURE_VULNERABLE_PENALTY 4000
#define DOUBLE_THREE_PENALTY 1800  // same as BLOCK_OPEN_2

// not used in hard eval
#define CONTINUOUS_LINE_4 10000
#define CONTINUOUS_LINE_3 100
#define CONTINUOUS_LINE_2 10
#define CONTINUOUS_LINE_1 1

// not used in hard eval
#define BLOCK_LINE_5 9990
#define BLOCK_LINE_4 6000
#define BLOCK_LINE_3 5000
#define BLOCK_LINE_2 4000
#define BLOCK_LINE_1 3000

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
  // attack
  int gomokuCount;
  int openFourCount;
  int closedFourCount;
  int openThreeCount;
  int closedThreeCount;
  int openTwoCount;
  int threatCount;
  int captureCount;

  // block
  int fourBlockCount;
  int gomokuBlockCount;
  int closedThreeBlockCount;
  int openThreeBlockCount;
  int openTwoBlockCount;
  int openOneBlockCount;

  // capture
  int captureVulnerable;
  int captureBlockCount;
  int captureThreatCount;
  int captureCriticalCount;
  int captureBlockCriticalCount;

  // critical
  int captureWin;
  int fixBreakableGomoku;
  int perfectCritical;

  PatternCounts()
      : gomokuCount(0),
        openFourCount(0),
        closedFourCount(0),
        openThreeCount(0),
        closedThreeCount(0),
        openTwoCount(0),
        threatCount(0),
        captureCount(0),
        fourBlockCount(0),
        gomokuBlockCount(0),
        closedThreeBlockCount(0),
        openThreeBlockCount(0),
        openTwoBlockCount(0),
        openOneBlockCount(0),
        captureVulnerable(0),
        captureBlockCount(0),
        captureThreatCount(0),
        captureCriticalCount(0),
        captureBlockCriticalCount(0),
        captureWin(0),
        fixBreakableGomoku(0),
        perfectCritical(0) {}
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
    counts.closedFourCount += other.counts.closedFourCount;
    counts.openThreeCount += other.counts.openThreeCount;
    counts.closedThreeCount += other.counts.closedThreeCount;
    counts.openTwoCount += other.counts.openTwoCount;
    counts.threatCount += other.counts.threatCount;
    counts.captureCount += other.counts.captureCount;
    counts.fourBlockCount += other.counts.fourBlockCount;
    counts.gomokuBlockCount += other.counts.gomokuBlockCount;
    counts.openThreeBlockCount += other.counts.openThreeBlockCount;
    counts.openTwoBlockCount += other.counts.openTwoBlockCount;
    counts.openOneBlockCount += other.counts.openOneBlockCount;
    counts.closedThreeBlockCount += other.counts.closedThreeBlockCount;
    counts.captureVulnerable += other.counts.captureVulnerable;
    counts.captureBlockCount += other.counts.captureBlockCount;
    counts.captureThreatCount += other.counts.captureThreatCount;
    counts.captureCriticalCount += other.counts.captureCriticalCount;
    counts.captureBlockCriticalCount += other.counts.captureBlockCriticalCount;
    counts.captureWin += other.counts.captureWin;
    counts.gomokuCount += other.counts.gomokuCount;
    counts.fixBreakableGomoku += other.counts.fixBreakableGomoku;
    counts.perfectCritical += other.counts.perfectCritical;
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

void printAxis(int forward, int backward);
bool isValidBackwardPattern(unsigned int sidePattern);
bool isValidForwardPattern(unsigned int sidePattern);
bool isCaptureWarning(int side, int player, bool reverse);
bool isCaptureVulnerable(int forward, int backward, int player);

void slideWindowContinuous(int side, int player, bool reverse, int &continuous, bool &isClosedEnd,
                           int &continuousEmpty, int &emptyThenContinuous,
                           int &emptyEmptyThenContinuous);

void initCombinedPatternScoreTables();
void initCombinedPatternScoreTablesHard();

int checkVPattern(Board *board, int player, int x, int y, int i);
int checkCapture(unsigned int side, unsigned int player);

unsigned int reversePattern(unsigned int pattern, int windowSize);

int evaluatePosition(Board *&board, int player, int x, int y);

int evaluatePositionHard(Board *&board, int player, int x, int y);

int getEvaluationPercentage(int score);

void printPattern(unsigned int pattern, int numCells);

// std::vector<std::pair<int, int> > getThresholdOpponentTotal(Board *board);

}  // namespace Evaluation
#endif  // EVALUATION_HPP
