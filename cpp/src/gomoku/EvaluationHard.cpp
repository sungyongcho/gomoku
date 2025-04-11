#include "Evaluation.hpp"
namespace Evaluation {
const int SCORE_IMMEDIATE_WIN = 1000000;          /// 100%
const int SCORE_IMMEDIATE_CAPTURE_WIN = 1000000;  /// 100%
const int SCORE_FORCED_WIN_SEQUENCE = 950000;     // 95%
const int SCORE_DOUBLE_OPEN_FOUR = 900000;        // 90%
const int SCORE_OPEN_FOUR = 700000;               // 70%
const int SCORE_CLOSED_FOUR = 400000;             // 40%
const int SCORE_DOUBLE_THREAT = 800000;           // 80%
const int SCORE_FORK = 750000;                    // 75%
const int SCORE_CAPTURE_LEADING = 500001;         // 50%
const int SCORE_OPPORTUNISTIC_CAPTURE = 300000;   // 30%
const int SCORE_OPEN_THREE = 200000;              // 20%
const int SCORE_CLOSED_THREE = 100000;            // 10%
const int SCORE_DEFENSIVE_BLOCK = 350000;         // 35%
const int SCORE_CHAIN_CAPTURE_SETUP = 250000;     // 25%
const int SCORE_COUNTER_THREAT = 300000;          // 30%
const int SCORE_POSITIONAL_ADVANTAGE = 150000;    // 15 %

EvaluationEntry evaluateContinuousPatternHard(unsigned int backward, unsigned int forward,
                                              unsigned int player) {
  EvaluationEntry returnValue;

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

  // if continous on both side, it'll be gomoku (5 in a row)
  if (totalContinuous >= 4) {
    // returnValue.counts.openFourCount += 1;
    totalContinuous = 4;
  }

  // prevent from placing unnecessary moves by setting total continuous 0
  if (totalContinuous < 4) {
    // 1. for condition where total continous are less than equal to
    // if both ends are closed, it is meaningless to place the stone.
    if (backwardClosedEnd == true && forwardClosedEnd == true) totalContinuous = 0;

    // 2. if the total continuous + continuous empty => potential growth for gomoku is less then
    // five, don't need to extend the line
    else if (!((totalContinuous + forwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty + forwardContinuousEmpty) >= 5))
      totalContinuous = 0;

    // 3. prevent from opponent to capture (needs to check if necessary)
    // separated if condition because it needs to check all above then add
    if (totalContinuous == 0 &&
        (isCaptureWarning(forward, player, false) || isCaptureWarning(backward, player, true)))
      totalContinuous = forwardContinuous + backwardContinuous;
  }

  if (isCaptureVulnerable(forward, backward, player)) totalContinuous = 0;

  if (totalContinuous >= 4) totalContinuous = 4;

  if (totalContinuous == 3) {
    returnValue.score += CONTINUOUS_LINE_4;
    if (!forwardClosedEnd && !backwardClosedEnd) returnValue.counts.openFourCount += 1;
    // else if (!forwardClosedEnd || !backwardClosedEnd)
    //   returnValue.counts.threatCount += 1;
  }

  if (totalContinuous == 2) {
    if (!forwardClosedEnd && !backwardClosedEnd) {
      // open three
      returnValue.counts.threatCount += 1;
    } else if (!forwardClosedEnd || !backwardClosedEnd) {
      // closed three
      returnValue.counts.threatCount += 1;
    }
  }

  int forwardBlockContinuous = 0;
  bool forwardBlockClosedEnd = false;

  int backwardBlockContinuous = 0;
  bool backwardBlockClosedEnd = false;
  slideWindowBlock(forward, player, false, forwardBlockContinuous, forwardBlockClosedEnd);
  slideWindowBlock(backward, player, true, backwardBlockContinuous, backwardBlockClosedEnd);

  int totalBlockCont = forwardBlockContinuous + backwardBlockContinuous;

  // if continous opponent is bigger or equal, should block asap
  if (totalBlockCont >= 4) totalBlockCont = 4;

  if (totalBlockCont < 4) {
    // if both end is blocked by player and continous is less then three, there is no need to
    // block
    if (forwardBlockClosedEnd && backwardBlockClosedEnd) totalBlockCont = 0;
    // for each side, if one side continous but that side is already closed,
    // it doesn't need to be blocked 'yet', so heuristics can go for better score moves.
    else if ((forwardBlockClosedEnd && (forwardBlockContinuous == totalBlockCont)) ||
             (backwardBlockClosedEnd && (backwardBlockContinuous == totalBlockCont))) {
      totalBlockCont = 0;
    }
  }

  // TODO add more condition
  if (totalBlockCont == 3 && !forwardBlockClosedEnd && !backwardBlockClosedEnd)
    returnValue.counts.defensiveBlockCount += 1;

  if (checkCapture(forward, player) > 0) returnValue.counts.captureCount += 1;
  if (checkCapture(reversePattern(backward, SIDE_WINDOW_SIZE), player) > 0)
    returnValue.counts.captureCount += 1;

  if (checkCapture(forward, player) < 0) returnValue.counts.captureBlockCount += 1;
  if (checkCapture(reversePattern(backward, SIDE_WINDOW_SIZE), player) < 0)
    returnValue.counts.captureBlockCount += 1;

  returnValue.score += (continuousScores[totalContinuous + 1] + blockScores[totalBlockCont + 1]);
  return returnValue;
}

void initCombinedPatternScoreTablesHard() {
  std::fill(patternPlayerOne, patternPlayerOne + LOOKUP_TABLE_SIZE, INVALID_ENTRY);
  std::fill(patternPlayerTwo, patternPlayerTwo + LOOKUP_TABLE_SIZE, INVALID_ENTRY);

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
      patternPlayerOne[pattern] = evaluateContinuousPatternHard(backward, forward, PLAYER_1);
      patternPlayerTwo[pattern] = evaluateContinuousPatternHard(backward, forward, PLAYER_2);
    }
  }
}

// void updatePatternCounts(int score, PatternCounts &pattern) {
//   if (score == SCORE_OPEN_FOUR) pattern.openFourCount++;

//   if (score == SCORE_OPEN_THREE) pattern.threatCount++;

//   if (score == SCORE_CLOSED_THREE) pattern.threatCount++;

//   if (score % SCORE_CAPTURE_LEADING == 0) pattern.captureCount += score / SCORE_CAPTURE_LEADING;

//   if (score == SCORE_DEFENSIVE_BLOCK) pattern.defensiveBlockCount++;
// }

EvaluationEntry evaluateCombinedAxisHard(Board *board, int player, int x, int y, int dx, int dy) {
  unsigned int forward = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  unsigned int backward = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  unsigned int revBackward = reversePattern(backward, SIDE_WINDOW_SIZE);

  unsigned int combined =
      (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) | (0 << (2 * SIDE_WINDOW_SIZE)) | forward;

  if (player == PLAYER_1)
    return patternPlayerOne[combined];
  else
    return patternPlayerTwo[combined];
}

int evaluatePositionHard(Board *&board, int player, int x, int y) {
  EvaluationEntry total;

  // if (board->getValueBit(x, y) == EMPTY_SPACE) return 0;

  for (int i = 0; i < 4; ++i) {
    total += evaluateCombinedAxisHard(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);
  }

  if (total.counts.openFourCount >= 2) {
    total.score += FORCED_WIN_SEQUENCE;
    total.score += DOUBLE_OPEN_FOUR;
  }

  if (total.counts.threatCount >= 2) total.score += DOUBLE_THREAT;

  if (total.counts.threatCount >= 3) total.score += FORK;

  // if (total.counts.captureCount >= 2) total.score += SCORE_CHAIN_CAPTURE_SETUP;

  if (total.counts.defensiveBlockCount >= 2) total.score += COUNTER_THREAT;

  // int boardCenter = BOARD_SIZE / 2;
  // int posBonus = SCORE_POSITIONAL_ADVANTAGE - (abs(x - boardCenter) + abs(y - boardCenter)) *
  // 1000; totalScore += posBonus;

  int activeCaptureScore = (player == board->getLastPlayer()) ? board->getLastPlayerScore()
                                                              : board->getNextPlayerScore();
  int opponentCaptureScore = (player == board->getLastPlayer()) ? board->getNextPlayerScore()
                                                                : board->getLastPlayerScore();
  if (total.counts.captureCount > 0) {
    if (activeCaptureScore + total.counts.captureCount >= board->getGoal()) return CAPTURE_WIN;
    total.score += static_cast<int>(continuousScores[2] *
                                    std::pow(10, (activeCaptureScore + total.counts.captureCount)));
  }

  if (total.counts.captureBlockCount > 0) {
    if (opponentCaptureScore + total.counts.captureBlockCount >= board->getGoal())
      return BLOCK_LINE_5;
    total.score +=
        static_cast<int>(continuousScores[2] *
                         std::pow(10, (opponentCaptureScore + total.counts.captureBlockCount)));
  }
  // totalScore += (activeCaptureScore - opponentCaptureScore) * SCORE_OPPORTUNISTIC_CAPTURE;

  return total.score;
}

}  // namespace Evaluation
