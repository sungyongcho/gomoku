#include "Evaluation.hpp"

#include <iostream>
namespace Evaluation {
int patternScoreTablePlayerOne[LOOKUP_TABLE_SIZE];
int patternScoreTablePlayerTwo[LOOKUP_TABLE_SIZE];

bool isValidBackwardPattern(unsigned int sidePattern) {
  bool encounteredValid = false;
  for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
    // Calculate shift: i=0 -> outermost cell (highest bits)
    int shift = 2 * (SIDE_WINDOW_SIZE - 1 - i);
    int cell = (sidePattern >> shift) & 0x3;
    if (!encounteredValid) {
      if (cell != 3) encounteredValid = true;
    } else {
      if (cell == 3) return false;
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
  // Check the cell closest to the center (highest order bits)
  int shift = 2 * (SIDE_WINDOW_SIZE - 1);
  int firstCell = (sidePattern >> shift) & 0x3;
  if (firstCell == 3) return false;

  bool encounteredOOB = false;
  for (int i = 1; i < SIDE_WINDOW_SIZE; ++i) {
    shift = 2 * (SIDE_WINDOW_SIZE - 1 - i);
    int cell = (sidePattern >> shift) & 0x3;
    if (!encounteredOOB) {
      if (cell == 3) encounteredOOB = true;
    } else {
      if (cell != 3) return false;
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
                           int &continuousEmpty) {
  int opponent = player == 1 ? 2 : 1;
  (void)reverse;
  int player_count = 0;
  int closed = 0;
  int player_begin = 0;

  if (!reverse) {
    for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
      int target_bit = ((side >> ((SIDE_WINDOW_SIZE - i - 1) * 2)) & 0x03);
      if (target_bit == player) {
        if (continuous == i) continuous++;
        if (player_begin == 0) player_begin = i + 1;
        player_count++;
      } else if (target_bit == opponent || target_bit == OUT_OF_BOUNDS) {
        if (closed == 0) closed = i + 1;
      }
    }
    for (int i = SIDE_WINDOW_SIZE - continuous; i > 0; i--) {
      if (((side >> ((i - 1) * 2)) & 0x03) == EMPTY_SPACE)
        continuousEmpty += 1;
      else
        break;
    }
  } else {
    for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
      int target_bit = ((side >> (i * 2)) & 0x03);
      if (target_bit == player) {
        if (continuous == i) continuous++;
        if (player_begin == 0) player_begin = i + 1;
        player_count++;
      } else if (target_bit == opponent || target_bit == OUT_OF_BOUNDS) {
        if (closed == 0) closed = i + 1;
      }
    }
    for (int i = continuous; i < SIDE_WINDOW_SIZE; i++) {
      if (((side >> (i * 2)) & 0x03) == EMPTY_SPACE)
        continuousEmpty += 1;
      else
        break;
    }
  }
  // std::cout << "----------------" << std::endl;
  // std::cout << "player_count: " << player_count << std::endl;
  // std::cout << "continuous: " << continuous << std::endl;
  // std::cout << "closed: " << closed << std::endl;
  // std::cout << "player_begin: " << player_begin << std::endl;
  // std::cout << "continuousEmpty: " << continuousEmpty << std::endl;
  // std::cout << "----------------" << std::endl;

  if (player_count == continuous) {
    if (closed - continuous == 1) {
      isClosedEnd = true;
    }
  }
}

void slideWindowBlock(int side, int player, bool reverse, int &blockContinuous, bool &isClosedEnd) {
  int opponent = player == 1 ? 2 : 1;
  int blockContinuousEmpty = 0;
  int closed = 0;

  if (!reverse) {
    for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
      int target_bit = ((side >> ((SIDE_WINDOW_SIZE - i - 1) * 2)) & 0x03);
      if (target_bit == opponent) {
        if (blockContinuous == i) blockContinuous++;
      } else if (target_bit == player || target_bit == OUT_OF_BOUNDS) {
        if (closed == 0) closed = i + 1;
      }
    }
    for (int i = SIDE_WINDOW_SIZE - blockContinuous; i > 0; i--) {
      if (((side >> ((i - 1) * 2)) & 0x03) == EMPTY_SPACE)
        blockContinuousEmpty += 1;
      else
        break;
    }
  } else {
    for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
      int target_bit = ((side >> (i * 2)) & 0x03);
      if (target_bit == opponent) {
        if (blockContinuous == i) blockContinuous++;
      } else if (target_bit == player || target_bit == OUT_OF_BOUNDS) {
        if (closed == 0) closed = i + 1;
      }
    }
    for (int i = blockContinuous; i < SIDE_WINDOW_SIZE; i++) {
      if (((side >> (i * 2)) & 0x03) == EMPTY_SPACE)
        blockContinuousEmpty += 1;
      else
        break;
    }
  }

  if (closed - blockContinuous == 1) {
    isClosedEnd = true;
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

bool isCaptureVulnerable(int forward, int backward, int player) {
  int opponent = player == 1 ? 2 : 1;

  if (((forward & 0xF0) >> 4) == pack_cells_2(player, opponent) &&
      ((backward & 0x03) == EMPTY_SPACE))
    return true;
  if (((forward & 0xC0) >> 6) == opponent &&
      ((backward & 0x0F) == pack_cells_2(EMPTY_SPACE, player)))
    return true;

  if (((backward & 0x0F) == pack_cells_2(opponent, player)) &&
      (((forward & 0xC0) >> 6) == EMPTY_SPACE))
    return true;

  if (((backward & 0x03) == opponent) &&
      (((forward & 0x0F) >> 4) == pack_cells_2(player, EMPTY_SPACE)))
    return true;

  return false;
}

int evaluateContinuousPattern(unsigned int backward, unsigned int forward, unsigned int player) {
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

  if (isCaptureVulnerable(forward, backward, player)) totalContinuous = 0;

  int forwardBlockContinuous = 0;
  bool forwardBlockClosedEnd = false;

  int backwardBlockContinuous = 0;
  bool backwardBlockClosedEnd = false;
  slideWindowBlock(forward, player, false, forwardBlockContinuous, forwardBlockClosedEnd);
  slideWindowBlock(backward, player, true, backwardBlockContinuous, backwardBlockClosedEnd);

  int totalBlockCont = forwardBlockContinuous + backwardBlockContinuous;
  // 1.if continous opponent is bigger or equal, should block asap
  if (totalBlockCont >= 4) totalBlockCont = 4;

  if (totalBlockCont < 4) {
    // 2. if both end is blocked by player and continous is less then three, there is no need to
    // block
    if (forwardBlockClosedEnd && backwardBlockClosedEnd) totalBlockCont = 0;
    // 3. for each side, if one side continous but that side is already closed,
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
    return CAPTURE_SCORE;
  else if (((side & 0xfc) >> 2) == pack_cells_3(player, player, opponent))
    return -CAPTURE_SCORE;
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
  unsigned int combined =
      (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) | (0 << (2 * SIDE_WINDOW_SIZE)) | forward;
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
      result += (continuousScores[2] * 2 + blockScores[2]) + 1;
  }
  return result;
}

int evaluatePosition(Board *&board, int player, int x, int y) {
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

int getEvaluationRating(int score) {
  // Map score from [1, GOMOKU] to [0, 100]
  int percentage = (score - 1) * 100 / (GOMOKU - 1);

  if (percentage < 20)
    return 1;
  else if (percentage < 40)
    return 2;
  else if (percentage < 60)
    return 3;
  else if (percentage < 80)
    return 4;
  else
    return 5;
}

}  // namespace Evaluation
