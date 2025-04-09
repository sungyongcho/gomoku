#include "Evaluation.hpp"
namespace Evaluation {
const int SCORE_IMMEDIATE_WIN = 1000000;
const int SCORE_IMMEDIATE_CAPTURE_WIN = 1000000;
const int SCORE_FORCED_WIN_SEQUENCE = 950000;
const int SCORE_DOUBLE_OPEN_FOUR = 900000;
const int SCORE_OPEN_FOUR = 700000;
const int SCORE_CLOSED_FOUR = 400000;
const int SCORE_DOUBLE_THREAT = 800000;
const int SCORE_FORK = 750000;
const int SCORE_CAPTURE_LEADING = 500001;
const int SCORE_OPPORTUNISTIC_CAPTURE = 300000;  // also used as capture board difference factor
const int SCORE_OPEN_THREE = 200000;
const int SCORE_CLOSED_THREE = 100000;
const int SCORE_DEFENSIVE_BLOCK = 350000;
const int SCORE_CHAIN_CAPTURE_SETUP = 250000;
const int SCORE_COUNTER_THREAT = 300000;
const int SCORE_POSITIONAL_ADVANTAGE = 150000;

struct PatternCounts {
  int openFourCount;        // Used for forced win / double open four detection.
  int threatCount;          // Counts open/closed three threats.
  int captureCount;         // Counts capture opportunities ("OXXO").
  int defensiveBlockCount;  // Counts when this move blocks an opponent pattern.

  PatternCounts() : openFourCount(0), threatCount(0), captureCount(0), defensiveBlockCount(0) {}
};

int evaluateContinuousPatternHard(unsigned int backward, unsigned int forward,
                                  unsigned int player) {
  int forwardContinuous = 0;
  bool forwardClosedEnd = false;
  int forwardContinuousEmpty = 0;

  int backwardContinuous = 0;
  bool backwardClosedEnd = false;
  int backwardContinuousEmpty = 0;

  slideWindowContinuous(forward, player, false, forwardContinuous, forwardClosedEnd,
                        forwardContinuousEmpty);
  slideWindowContinuous(backward, player, true, backwardContinuous, backwardClosedEnd,
                        backwardContinuousEmpty);

  int totalContinuous = forwardContinuous + backwardContinuous;

  if (totalContinuous >= 4) return SCORE_IMMEDIATE_WIN;

  if (totalContinuous == 3) {
    if (!forwardClosedEnd && !backwardClosedEnd)
      return SCORE_OPEN_FOUR;
    else if (!forwardClosedEnd || !backwardClosedEnd)
      return SCORE_CLOSED_FOUR;
  }

  if (totalContinuous == 2) {
    if (!forwardClosedEnd && !backwardClosedEnd)
      return SCORE_OPEN_THREE;
    else if (!forwardClosedEnd || !backwardClosedEnd)
      return SCORE_CLOSED_THREE;
  }

  int captureCount = 0;

  if (checkCapture(forward, player) > 0) captureCount++;
  if (checkCapture(reversePattern(backward, SIDE_WINDOW_SIZE), player) > 0) captureCount++;

  if (captureCount > 0) return captureCount * SCORE_CAPTURE_LEADING;

  int forwardBlockContinuous = 0;
  bool forwardBlockClosedEnd = false;

  int backwardBlockContinuous = 0;
  bool backwardBlockClosedEnd = false;
  slideWindowBlock(forward, player, false, forwardBlockContinuous, forwardBlockClosedEnd);
  slideWindowBlock(backward, player, true, backwardBlockContinuous, backwardBlockClosedEnd);

  int totalBlockCont = forwardBlockContinuous + backwardBlockContinuous;

  if (totalBlockCont == 3 && !forwardBlockClosedEnd && !backwardBlockClosedEnd)
    return SCORE_DEFENSIVE_BLOCK;

  return 0;
}

void initCombinedPatternScoreTablesHard() {
  std::fill(patternScoreTablePlayerTwo, patternScoreTablePlayerOne + LOOKUP_TABLE_SIZE,
            INVALID_PATTERN);
  std::fill(patternScoreTablePlayerTwo, patternScoreTablePlayerTwo + LOOKUP_TABLE_SIZE,
            INVALID_PATTERN);

  const unsigned int sideCount = 1 << (2 * SIDE_WINDOW_SIZE);

  // Iterate over all possible left and right side patterns.
  for (unsigned int backward = 0; backward < sideCount; ++backward) {
    if (!isValidBackwardPattern(backward)) continue;
    for (unsigned int forward = 0; forward < sideCount; ++forward) {
      if (!isValidForwardPattern(forward)) continue;

      // Build the full pattern:
      // Left side occupies the highest 2*SIDE_WINDOW_SIZE bits,
      // then the fixed center (2 bits),
      // and then the forward side occupies the lowest 2*SIDE_WINDOW_SIZE bits.
      unsigned int pattern = (backward << (2 * (SIDE_WINDOW_SIZE + 1))) |
                             (WINDOW_CENTER_VALUE << (2 * SIDE_WINDOW_SIZE)) | forward;
      patternScoreTablePlayerOne[pattern] =
          evaluateContinuousPatternHard(backward, forward, PLAYER_1);
      patternScoreTablePlayerTwo[pattern] =
          evaluateContinuousPatternHard(backward, forward, PLAYER_2);
    }
  }
}

void updatePatternCounts(int score, PatternCounts &pattern) {
  if (score == SCORE_OPEN_FOUR) pattern.openFourCount++;

  if (score == SCORE_OPEN_THREE) pattern.threatCount++;

  if (score == SCORE_CLOSED_THREE) pattern.threatCount++;

  if (score % SCORE_CAPTURE_LEADING == 0) pattern.captureCount += score / SCORE_CAPTURE_LEADING;

  if (score == SCORE_DEFENSIVE_BLOCK) pattern.defensiveBlockCount++;
}

int evaluateCombinedAxis(Board *board, int player, int x, int y, int dx, int dy,
                         PatternCounts &pattern) {
  int score = 0;
  unsigned int forward = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  unsigned int backward = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  unsigned int revBackward = reversePattern(backward, SIDE_WINDOW_SIZE);

  unsigned int combined =
      (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) | (0 << (2 * SIDE_WINDOW_SIZE)) | forward;

  if (player == PLAYER_1) {
    score = patternScoreTablePlayerOne[combined];
  } else if (player == PLAYER_2) {
    score = patternScoreTablePlayerTwo[combined];
  }

  updatePatternCounts(score, pattern);

  return score;
}

int evaluatePositionHard(Board *&board, int player, int x, int y) {
  PatternCounts pattern;

  int totalScore = 0;
  // if (board->getValueBit(x, y) == EMPTY_SPACE) return 0;

  for (int i = 0; i < 4; ++i) {
    totalScore +=
        evaluateCombinedAxis(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1], pattern);
  }

  if (pattern.openFourCount >= 2) {
    totalScore += SCORE_FORCED_WIN_SEQUENCE;
    totalScore += SCORE_DOUBLE_OPEN_FOUR;
  }

  if (pattern.threatCount >= 2) totalScore += SCORE_DOUBLE_THREAT;

  if (pattern.threatCount >= 3) totalScore += SCORE_FORK;

  if (pattern.captureCount >= 2) totalScore += SCORE_CHAIN_CAPTURE_SETUP;

  if (pattern.defensiveBlockCount >= 2) totalScore += SCORE_COUNTER_THREAT;

  int boardCenter = BOARD_SIZE / 2;
  int posBonus = SCORE_POSITIONAL_ADVANTAGE - (abs(x - boardCenter) + abs(y - boardCenter)) * 1000;
  totalScore += posBonus;

  int activeCaptureScore = (player == board->getLastPlayer()) ? board->getLastPlayerScore()
                                                              : board->getNextPlayerScore();
  int opponentCaptureScore = (player == board->getLastPlayer()) ? board->getNextPlayerScore()
                                                                : board->getLastPlayerScore();
  totalScore += (activeCaptureScore - opponentCaptureScore) * SCORE_OPPORTUNISTIC_CAPTURE;

  return totalScore;
}

}  // namespace Evaluation
