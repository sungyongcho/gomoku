#include "Evaluation.hpp"

#include <iostream>
namespace Evaluation {
int patternScoreTablePlayerOne[LOOKUP_TABLE_SIZE];
int patternScoreTablePlayerTwo[LOOKUP_TABLE_SIZE];

void printAxis(int forward, int backward) {
  // Process backward 8 bits in 2-bit groups (from MSB to LSB)
  for (int i = 3; i >= 0; i--) {
    int val = (backward >> (i * 2)) & 0x03;
    switch (val) {
      case 0:
        std::cout << ".";
        break;
      case 1:
        std::cout << "1";
        break;
      case 2:
        std::cout << "2";
        break;
      case 3:
        std::cout << "X";
        break;
    }
  }
  // Print the middle marker "[.]"
  std::cout << "[.]";

  // Process forward 8 bits in 2-bit groups (from MSB to LSB)
  for (int i = 3; i >= 0; i--) {
    int val = (forward >> (i * 2)) & 0x03;
    switch (val) {
      case 0:
        std::cout << ".";
        break;
      case 1:
        std::cout << "1";
        break;
      case 2:
        std::cout << "2";
        break;
      case 3:
        std::cout << "X";
        break;
    }
  }
  std::cout << std::endl;
}

void printCombined(unsigned int combined) {
  for (int i = 8; i >= 0; i--) {
    int val = (combined >> (i * 2)) & 0x03;
    if (i == 4) {
      std::cout << "[.]";
      continue;
    }
    switch (val) {
      case 0:
        std::cout << ".";
        break;
      case 1:
        std::cout << "1";
        break;
      case 2:
        std::cout << "2";
        break;
      case 3:
        std::cout << "X";
        break;
    }
  }
  std::cout << std::endl;
}

bool isValidBackwardPattern(unsigned int sidePattern) {
  bool encounteredValid = false;

  for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
    int shift = 2 * (SIDE_WINDOW_SIZE - 1 - i);
    int cell = (sidePattern >> shift) & 0x3;

    if (encounteredValid) {
      if (cell == 3) return false;
    } else {
      if (cell != 3) encounteredValid = true;
    }
  }

  return true;
}

// Right side validation:
// The right side (SIDE_WINDOW_SIZE cells) is represented so that the cell closest to the center
// is in the highest order bits. Validity rule:
// - The cell closest to the center must not be OUT_OF_BOUNDS (3).
// - After an OUT_OF_BOUNDS appears, all further cells (moving outward) must be 3.
bool isValidForwardPattern(unsigned int sidePattern) {
  bool encounteredOOB = false;

  for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
    int shift = 2 * (SIDE_WINDOW_SIZE - 1 - i);
    int cell = (sidePattern >> shift) & 0x3;

    if (cell == 3) {
      encounteredOOB = true;
    } else {
      if (encounteredOOB) return false;
    }
  }

  return true;
}

// to remove
void printPattern(unsigned int pattern, int numCells) {
  for (int i = 0; i < numCells; ++i) {
    // Calculate shift so that the leftmost cell is printed first.
    int shift = 2 * (numCells - 1 - i);
    int cell = (pattern >> shift) & 0x3;
    std::cout << cell << " ";
  }
  std::cout << std::endl;
}

void slideWindowContinuous(int side, int player, bool reverse, int &continuous, bool &isClosedEnd,
                           int &continuousEmpty, int &emptyThenContinuous,
                           int &emptyEmptyThenContinuous) {
  int opponent = player == 1 ? 2 : 1;
  int emptyPassed = 0;
  side = reverse ? reversePattern(side, SIDE_WINDOW_SIZE) : side;
  for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
    int target_bit = ((side >> ((SIDE_WINDOW_SIZE - i - 1) * 2)) & 0x03);
    if (target_bit == player) {
      if (continuous == i && !isClosedEnd) continuous++;
      if (emptyPassed == 1 && emptyThenContinuous == (i - 1) && !isClosedEnd) emptyThenContinuous++;
      if (emptyPassed == 2 && emptyEmptyThenContinuous == (i - 2) && !isClosedEnd)
        emptyEmptyThenContinuous++;
    } else if (!emptyPassed && (target_bit == opponent || target_bit == OUT_OF_BOUNDS)) {
      isClosedEnd = true;
    } else if (target_bit == EMPTY_SPACE && !emptyPassed) {
      emptyPassed += 1;
    }
  }
  for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
    int target_bit = ((side >> ((SIDE_WINDOW_SIZE - i - 1) * 2)) & 0x03);
    if (target_bit == EMPTY_SPACE) continuousEmpty += 1;
    if (target_bit == player) continue;
    if (target_bit == opponent || target_bit == OUT_OF_BOUNDS) break;
  }
}

bool isCaptureWarning(int side, int player, bool reverse) {
  int opponent = player == 1 ? 2 : 1;

  // check forward
  if (!reverse) {
    if (((side & 0xFC) >> 2) == pack_cells_3(player, player, opponent)) return true;
  } else {
    if ((side & 0x3F) == pack_cells_3(opponent, player, player)) return true;
  }
  return false;
}

int getCell(int bits, int index) {
  int shift = 2 * (SIDE_WINDOW_SIZE - index - 1);  // 각 셀은 2비트
  return (bits >> shift) & 0x03;
}

// Check the player has risk of capture if he plays at (x, y)
bool isCaptureVulnerable(int forward, int backward, int player) {
  int opponent = player == 1 ? 2 : 1;

  // BACKWARD 기반 패턴
  int B1 = getCell(backward, 2);
  int B0 = getCell(backward, 3);
  int F0 = getCell(forward, 0);
  int F1 = getCell(forward, 1);

  // 패턴 1: opponent - player - [P] - empty
  if (B1 == opponent && B0 == player && F0 == EMPTY_SPACE) {
    return true;
  }

  // 패턴 2: opponent - [P] - player - empty
  if (B0 == opponent && F0 == player && F1 == EMPTY_SPACE) {
    return true;
  }

  // 패턴 3: empty - [P] - player - opponent
  if (B0 == EMPTY_SPACE && F0 == player && F1 == opponent) {
    return true;
  }

  // 패턴 4: empty - player - [P] - opponent
  if (B1 == EMPTY_SPACE && B0 == player && F0 == opponent) {
    return true;
  }

  return false;
}

int evaluateContinuousPattern(unsigned int backward, unsigned int forward, unsigned int player) {
  int forwardContinuous = 0;
  bool forwardClosedEnd = false;
  int forwardContinuousEmpty = 0;
  int forwardEmptyThenContinuous = 0;
  int forwardEmptyEmptyThenContinuous = 0;

  int backwardContinuous = 0;
  bool backwardClosedEnd = false;
  int backwardContinuousEmpty = 0;
  int backwardEmptyThenContinuous = 0;
  int backwardEmptyEmptyThenContinuous = 0;
  slideWindowContinuous(forward, player, false, forwardContinuous, forwardClosedEnd,
                        forwardContinuousEmpty, forwardEmptyThenContinuous,
                        forwardEmptyEmptyThenContinuous);
  slideWindowContinuous(backward, player, true, backwardContinuous, backwardClosedEnd,
                        backwardContinuousEmpty, backwardEmptyThenContinuous,
                        backwardEmptyEmptyThenContinuous);

  int totalContinuous = forwardContinuous + backwardContinuous;

  // 1.if continous on both side, it'll be gomoku (5 in a row)
  if (totalContinuous >= 4) totalContinuous = 4;

  if (totalContinuous < 4) {
    // 2. for condition where total continous are less than equal to
    // if both ends are closed, it is meaningless to place the stone.
    if (backwardClosedEnd == true && forwardClosedEnd == true) totalContinuous = 0;

    // 3. if the total continuous + continuous empty => potential growth for gomoku is less then
    // five, don't need to extend the line
    else if (!((totalContinuous + forwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty + forwardContinuousEmpty) >= 5))
      totalContinuous = 0;

    // 4. prevent from opponent to capture (needs to check if necessary)
    // separated if condition because it needs to check all above then add
    if (totalContinuous == 0 &&
        (isCaptureWarning(forward, player, false) || isCaptureWarning(backward, player, true)))
      totalContinuous = forwardContinuous + backwardContinuous;
  }

  int opponent = player == 1 ? 2 : 1;
  int forwardBlockContinuous = 0;
  int forwardBlockContinuousEmpty = 0;
  bool forwardBlockClosedEnd = false;
  int forwardBlockEmptyThenContinuous = 0;
  int forwardBlockEmptyEmptyThenContinuous = 0;

  int backwardBlockContinuous = 0;
  int backwardBlockContinuousEmpty = 0;
  bool backwardBlockClosedEnd = false;
  int backwardBlockEmptyThenContinuous = 0;
  int backwardBlockEmptyEmptyThenContinuous = 0;

  slideWindowContinuous(forward, opponent, false, forwardBlockContinuous, forwardBlockClosedEnd,
                        forwardBlockContinuousEmpty, forwardBlockEmptyThenContinuous,
                        forwardBlockEmptyEmptyThenContinuous);
  slideWindowContinuous(backward, opponent, true, backwardBlockContinuous, backwardBlockClosedEnd,
                        backwardBlockContinuousEmpty, backwardBlockEmptyThenContinuous,
                        backwardBlockEmptyEmptyThenContinuous);

  int totalBlockCont = forwardBlockContinuous + backwardBlockContinuous;

  // 1.if continuous opponent is bigger or equal, should block asap
  if (totalBlockCont >= 4) totalBlockCont = 4;

  if (totalBlockCont < 4) {
    // 2. if both end is blocked by player and continuous is less then three, there is no need to
    // block
    if (forwardBlockClosedEnd && backwardBlockClosedEnd) totalBlockCont = 0;
    // 3. for each side, if one side continuous but that side is already closed,
    // it doesn't need to be blocked 'yet', so heuristics can go for better score moves.
    else if ((forwardBlockClosedEnd && (forwardBlockContinuous == totalBlockCont)) ||
             (backwardBlockClosedEnd && (backwardBlockContinuous == totalBlockCont))) {
      totalBlockCont = 0;
      // 3-2. but if it can be captured, add up the score (check)
      if (forwardBlockContinuous == 2) totalBlockCont += 1;
      if (backwardBlockContinuous == 2) totalBlockCont += 1;
    }
  }

  return continuousScores[totalContinuous + 1] + blockScores[totalBlockCont + 1];
}

void initCombinedPatternScoreTables() {
  std::fill(patternScoreTablePlayerOne, patternScoreTablePlayerOne + LOOKUP_TABLE_SIZE,
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
      patternScoreTablePlayerOne[pattern] = evaluateContinuousPattern(backward, forward, PLAYER_1);
      patternScoreTablePlayerTwo[pattern] = evaluateContinuousPattern(backward, forward, PLAYER_2);
    }
  }
}

unsigned int reversePattern(unsigned int pattern, int windowSize) {
  unsigned int reversed = 0;
  for (int i = 0; i < windowSize; i++) {
    reversed = (reversed << 2) | (pattern & 0x3);
    pattern >>= 2;
  }
  return reversed;
}

int checkCapture(unsigned int side, unsigned int player) {
  unsigned int opponent = OPPONENT(player);
  if (((side & 0xFC) >> 2) == pack_cells_3(opponent, opponent, player))
    return CAPTURE;
  else if (((side & 0xfc) >> 2) == pack_cells_3(player, player, opponent))
    return -CAPTURE;
  return 0;
}

int evaluateCombinedAxis(Board *board, int player, int x, int y, int dx, int dy) {
  int score = 0;
  // Extract windows.
  unsigned int forward = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  unsigned int backward = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  unsigned int revBackward = reversePattern(backward, SIDE_WINDOW_SIZE);
  // Combine: [reversed backward window] + [center cell (player)] + [forward
  // window]
  // unsigned int combined = (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) |
  //                         ((unsigned int)player << (2 * SIDE_WINDOW_SIZE)) | forward;
  unsigned int combined = (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) |
                          (WINDOW_CENTER_VALUE << (2 * SIDE_WINDOW_SIZE)) | forward;
  if (player == PLAYER_1) {
    score = patternScoreTablePlayerOne[combined];
  } else if (player == PLAYER_2) {
    score = patternScoreTablePlayerTwo[combined];
  }

  int activeCaptureScore = (player == board->getLastPlayer()) ? board->getLastPlayerScore()
                                                              : board->getNextPlayerScore();
  int opponentCaptureScore = (player == board->getLastPlayer()) ? board->getNextPlayerScore()
                                                                : board->getLastPlayerScore();
  // double goalRatio = board->getGoal();

  if (checkCapture(forward, player) > 0) activeCaptureScore++;
  if (checkCapture(backward, player) > 0) activeCaptureScore++;

  if (checkCapture(forward, player) < 0) opponentCaptureScore++;
  if (checkCapture(backward, player) < 0) opponentCaptureScore++;

  activeCaptureScore = activeCaptureScore / 2 + 1;
  opponentCaptureScore = opponentCaptureScore / 2 + 1;

  if (checkCapture(forward, player) > 0 || checkCapture(backward, player) > 0) {
    if (activeCaptureScore == board->getGoal()) return GOMOKU;
    score += static_cast<int>(continuousScores[2] * std::pow(10, (activeCaptureScore + 1)));
  } else if (checkCapture(forward, player) < 0 || checkCapture(backward, player) < 0) {
    if (opponentCaptureScore == board->getGoal()) return GOMOKU - 1;
    score += static_cast<int>(blockScores[2] * std::pow(10, (opponentCaptureScore + 1)));
  }

  return score;
}

int checkVPattern(Board *board, int player, int x, int y, int i) {
  int result = 0;
  int opponent = OPPONENT(player);

  // int i = begin % 8;

  int right = board->getValueBit(x + DIRECTIONS[i % 8][0], y + DIRECTIONS[i % 8][1]);
  int center = board->getValueBit(x + DIRECTIONS[(i + 1) % 8][0], y + DIRECTIONS[(i + 1) % 8][1]);
  int left = board->getValueBit(x + DIRECTIONS[(i + 2) % 8][0], y + DIRECTIONS[(i + 2) % 8][1]);

  if (!(right == opponent && left == opponent)) return result;
  // std::cout << right << " " << center << " " << left << std::endl;

  if (right == opponent && left == opponent) {
    if (center == EMPTY_SPACE || center == opponent)
      result += (continuousScores[2] * 3 + blockScores[2] * 3) + 1;
  }
  return result;
}

int evaluatePosition(Board *board, int player, int x, int y) {
  int totalScore = 0;

  // if (board->getValueBit(x, y) == EMPTY_SPACE) return 0;

  for (int i = 0; i < 4; ++i) {
    totalScore += evaluateCombinedAxis(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);
    if (totalScore >= GOMOKU) return totalScore;
  }
  for (int i = 1; i < 8; i += 2) {
    totalScore += checkVPattern(board, player, x, y, i);
  }
  return totalScore;
}

int getEvaluationPercentage(int score) {
  // Map score from [1, GOMOKU] to [0, 100]
  int percentage = (score - 1) * 100 / (GOMOKU - 1);

  return percentage;
}

}  // namespace Evaluation
