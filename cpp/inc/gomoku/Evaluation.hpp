#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <algorithm>
#include <cmath>
#include <set>

#include "Board.hpp"
#include "Gomoku.hpp"

#define CAPTURE_WIN 1100000
#define GOMOKU 1000000
#define CAPTURE_CRITICAL 1000000
#define DEFENSIVE_BLOCK 350000

#define CONTINUOUS_OPEN_4 900000
#define CONTINUOUS_CLOSED_4 500000
#define CONTINUOUS_OPEN_3 450000
#define CONTINUOUS_CLOSED_3 450000  // It blocks a capture

#define PERFECT_CRITICAL_LINE 900000
#define BLOCK_CRITICAL_LINE 900000
#define CONTINUOUS_LINE_4 1000000
#define CONTINUOUS_LINE_3 1000
#define CONTINUOUS_LINE_2 100
#define CONTINUOUS_LINE_1 10
#define BLOCK_LINE_5 999000
#define BLOCK_LINE_4 10000
#define BLOCK_LINE_3 100
#define BLOCK_LINE_2 10
#define BLOCK_LINE_1 1

#define CAPTURE 35000
#define THREAT 20000
#define THREAT_BLOCK 30000
#define CENTER_BONUS 10000
#define WINDOW_CENTER_VALUE 0
// TODO needs to check
#define INVALID_PATTERN -1337
#define CAPTURE_VULNERABLE_PENALTY 40000

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
  int gomokuCount;
  int openFourCount;          // Used for forced win / double open four detection.
  int closedFourCount;        // Used for forced win / double open four detection.
  int openThreeCount;         // Used for forced win / double open four detection.
  int closedThreeCount;       // Used for forced win / double open four detection.
  int threatCount;            // Counts open/closed three threats.
  int captureCount;           // Counts capture opportunities ("OXXO").
  int defensiveBlockCount;    // Counts when this move blocks an opponent pattern.
  int openFourBlockCount;     // Used for forced win / double open four detection.
  int closedFourBlockCount;   // Used for forced win / double open four detection.
  int openThreeBlockCount;    // Used for forced win / double open four detection.
  int closedThreeBlockCount;  // Used for forced win / double open four detection.
  int captureVulnerable;
  int captureBlockCount;
  int captureThreatCount;
  int captureCriticalCount;
  int captureWin;
  int fixBreakableGomoku;
  int perfectCritical;

  PatternCounts()
      : gomokuCount(0),
        openFourCount(0),
        closedFourCount(0),
        openThreeCount(0),
        closedThreeCount(0),
        threatCount(0),
        captureCount(0),
        defensiveBlockCount(0),
        openFourBlockCount(0),
        closedFourBlockCount(0),
        openThreeBlockCount(0),
        closedThreeBlockCount(0),
        captureVulnerable(0),
        captureBlockCount(0),
        captureThreatCount(0),
        captureCriticalCount(0),
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
    counts.threatCount += other.counts.threatCount;
    counts.captureCount += other.counts.captureCount;
    counts.defensiveBlockCount += other.counts.defensiveBlockCount;
    counts.openFourBlockCount += other.counts.openFourBlockCount;
    counts.closedFourBlockCount += other.counts.closedFourBlockCount;
    counts.openThreeBlockCount += other.counts.openThreeBlockCount;
    counts.closedThreeBlockCount += other.counts.closedThreeBlockCount;
    counts.captureVulnerable += other.counts.captureVulnerable;
    counts.captureBlockCount += other.counts.captureBlockCount;
    counts.captureThreatCount += other.counts.captureThreatCount;
    counts.captureCriticalCount += other.counts.captureCriticalCount;
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
void slideWindowBlock(int side, int player, bool reverse, int &blockContinuous, bool &isClosedEnd,
                      int &emptyThenContinuous);

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
