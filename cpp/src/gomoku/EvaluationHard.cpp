#include "Evaluation.hpp"

namespace Evaluation {

void printEvalEntry(EvaluationEntry eval) {
  std::cout << "=== EvalEntry ===" << std::endl;
  std::cout << "Score: " << eval.score << std::endl;
  if (eval.counts.openFourCount)
    std::cout << "openFourCount: " << eval.counts.openFourCount << std::endl;
  if (eval.counts.closedFourCount)
    std::cout << "closedFourCount: " << eval.counts.closedFourCount << std::endl;
  if (eval.counts.openThreeCount)
    std::cout << "openThreeCount: " << eval.counts.openThreeCount << std::endl;
  if (eval.counts.closedThreeCount)
    std::cout << "closedThreeCount: " << eval.counts.closedThreeCount << std::endl;
  if (eval.counts.openTwoCount)
    std::cout << "openTwoCount: " << eval.counts.openTwoCount << std::endl;
  if (eval.counts.threatCount) std::cout << "threatCount: " << eval.counts.threatCount << std::endl;
  if (eval.counts.captureCount)
    std::cout << "captureCount: " << eval.counts.captureCount << std::endl;
  if (eval.counts.fourBlockCount)
    std::cout << "fourBlockCount: " << eval.counts.fourBlockCount << std::endl;
  if (eval.counts.gomokuBlockCount)
    std::cout << "gomokuBlockCount: " << eval.counts.gomokuBlockCount << std::endl;
  if (eval.counts.openThreeBlockCount)
    std::cout << "openThreeBlockCount: " << eval.counts.openThreeBlockCount << std::endl;
  if (eval.counts.openTwoBlockCount)
    std::cout << "openTwoBlockCount: " << eval.counts.openTwoBlockCount << std::endl;
  if (eval.counts.closedThreeBlockCount)
    std::cout << "closedThreeBlockCount: " << eval.counts.closedThreeBlockCount << std::endl;
  if (eval.counts.openOneBlockCount)
    std::cout << "openOneBlockCount: " << eval.counts.openOneBlockCount << std::endl;
  if (eval.counts.captureVulnerable)
    std::cout << "captureVulnerable: " << eval.counts.captureVulnerable << std::endl;
  if (eval.counts.captureBlockCount)
    std::cout << "captureBlockCount: " << eval.counts.captureBlockCount << std::endl;
  if (eval.counts.captureThreatCount)
    std::cout << "captureThreatCount: " << eval.counts.captureThreatCount << std::endl;
  if (eval.counts.captureCriticalCount)
    std::cout << "captureCriticalCount: " << eval.counts.captureCriticalCount << std::endl;
  if (eval.counts.captureBlockCriticalCount)
    std::cout << "captureBlockCriticalCount: " << eval.counts.captureBlockCriticalCount
              << std::endl;
  if (eval.counts.captureWin) std::cout << "captureWin: " << eval.counts.captureWin << std::endl;
  if (eval.counts.gomokuCount) std::cout << "gomokuCount: " << eval.counts.gomokuCount << std::endl;
  if (eval.counts.fixBreakableGomoku)
    std::cout << "fixBreakableGomoku: " << eval.counts.fixBreakableGomoku << std::endl;
  if (eval.counts.perfectCritical)
    std::cout << "perfectCritical: " << eval.counts.perfectCritical << std::endl;
  std::cout << "=================" << std::endl;
}

unsigned int extractLineAsBitsFromBoard(Board* board, int x, int y, int dx, int dy) {
  unsigned int forwardBits = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  unsigned int backwardBits = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  unsigned int reversedBackwardBits = reversePattern(backwardBits, SIDE_WINDOW_SIZE);

  return (reversedBackwardBits << (2 * (SIDE_WINDOW_SIZE + 1))) |
         (WINDOW_CENTER_VALUE << (2 * SIDE_WINDOW_SIZE)) | forwardBits;
}

// It evaluates Continuous player's pattern & opponent's pattern
// Executed when server starts, create evaluation table
EvaluationEntry evaluateContinuousPatternHard(unsigned int backward, unsigned int forward,
                                              unsigned int player) {
  EvaluationEntry returnValue;

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

  // No need to extend after making gomoku, so set total continuous to 0
  if (totalContinuous >= 5) {
    totalContinuous = 5;
  }

  // prevent from placing unnecessary moves by setting total continuous 0
  if (totalContinuous < 4) {
    // 1. for condition where total continuous are less than equal to
    // if both ends are closed, it is meaningless to place the stone.
    if (backwardClosedEnd == true && forwardClosedEnd == true) {
      totalContinuous = 0;
    }
    // 2. if the total continuous + continuous empty => potential growth for gomoku is less then
    // five, don't need to extend the line
    else if (!((totalContinuous + forwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty + forwardContinuousEmpty) >= 5)) {
      totalContinuous = 0;
    }
  }

  if (totalContinuous == 4) {
    returnValue.counts.threatCount += 1;
    returnValue.counts.gomokuCount += 1;
  }

  if (totalContinuous == 3) {
    if (!forwardClosedEnd && !backwardClosedEnd) {
      returnValue.counts.openFourCount += 1;
      returnValue.counts.threatCount += 1;
    } else if ((!forwardClosedEnd && backwardClosedEnd) ||
               (forwardClosedEnd && !backwardClosedEnd)) {
      returnValue.counts.threatCount += 1;
      returnValue.counts.closedFourCount += 1;
    }
  }

  if (totalContinuous == 1) {
    if (!forwardClosedEnd && !backwardClosedEnd) {
      returnValue.counts.openTwoCount += 1;
    }
  }

  // [.]_OO_
  if (((forwardEmptyThenContinuous == 2) || (backwardEmptyThenContinuous == 2)) &&
      (!forwardClosedEnd && !backwardClosedEnd)) {
    returnValue.counts.threatCount += 1;
    returnValue.counts.openThreeCount += 1;
  }
  // 0[.]_0, 0_[.]0
  if (((forwardContinuous == 1 && backwardEmptyThenContinuous == 1) ||
       (forwardEmptyThenContinuous == 1 && backwardContinuous == 1)) &&
      (!forwardClosedEnd && !backwardClosedEnd)) {
    returnValue.counts.threatCount += 1;
    returnValue.counts.openThreeCount += 1;
  }

  if (totalContinuous == 2) {
    if (!forwardClosedEnd && !backwardClosedEnd) {
      // open three
      returnValue.counts.threatCount += 1;
      returnValue.counts.openThreeCount += 1;
    } else if (!forwardClosedEnd || !backwardClosedEnd) {
      // closed three
      // returnValue.counts.threatCount += 1;
      returnValue.counts.closedThreeCount += 1;
    }
  }

  if (isCaptureVulnerable(forward, backward, player)) {
    returnValue.counts.captureVulnerable += 1;
  }

  // -------------- Opponent Pattern count (Block) ----------- //
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

  int totalBlockContinuous = forwardBlockContinuous + backwardBlockContinuous;

  // if continuous opponent is bigger or equal, should block asap
  if (totalBlockContinuous >= 4) {
    totalBlockContinuous = 4;
  }
  if (totalBlockContinuous == 4) {
    if (!(forwardBlockClosedEnd && backwardBlockClosedEnd)) {
      returnValue.counts.fourBlockCount += 1;
    }
  }

  if (totalBlockContinuous == 3) {
    if (!forwardBlockClosedEnd && !backwardBlockClosedEnd) {
      returnValue.counts.openThreeBlockCount += 1;
    } else {
      returnValue.counts.closedThreeBlockCount += 1;
    }
  }

  // Block Open Two
  if (totalBlockContinuous == 2) {
    if (!forwardBlockClosedEnd && !backwardBlockClosedEnd) {
      returnValue.counts.openTwoBlockCount += 1;
    }
  }

  // Evaluate possible threat captures (_XX_)
  if (!forwardBlockClosedEnd && forwardBlockContinuous == 2) {
    returnValue.counts.captureThreatCount += 1;
  }
  if (!backwardBlockClosedEnd && backwardBlockContinuous == 2) {
    returnValue.counts.captureThreatCount += 1;
  }

  // Evaluate open two block
  if (totalBlockContinuous == 1) {
    if (!forwardBlockClosedEnd && !backwardBlockClosedEnd) {
      returnValue.counts.openOneBlockCount += 1;
    }
  }

  // Check player captures (OXX_) cases
  if (checkCapture(forward, player) > 0) returnValue.counts.captureCount += 1;
  if (checkCapture(reversePattern(backward, SIDE_WINDOW_SIZE), player) > 0)
    returnValue.counts.captureCount += 1;

  // Check opponent capture (XOO_) cases
  if (checkCapture(forward, player) < 0) returnValue.counts.captureBlockCount += 1;
  if (checkCapture(reversePattern(backward, SIDE_WINDOW_SIZE), player) < 0)
    returnValue.counts.captureBlockCount += 1;

  returnValue.score += blockScores[totalBlockContinuous + 1];
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

EvaluationEntry evaluateCombinedAxisHard(Board* board, int player, int x, int y, int dx, int dy) {
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

// Check if there is a Gomoku on a closed three pattern
static bool hasGomokuOnClosedThree(Board* board, int x, int y, int dx, int dy, int player) {
  int checkX = x;
  int checkY = y;

  for (int j = 0; j < 2; ++j) {
    checkX += dx;
    checkY += dy;
    if (board->getValueBit(checkX, checkY) != player) break;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);

      EvaluationEntry evaluation =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (evaluation.counts.gomokuCount > 0) {
        return true;
      }
    }
  }

  checkX = x;
  checkY = y;
  for (int j = 0; j < 2; ++j) {
    checkX -= dx;
    checkY -= dy;
    if (board->getValueBit(checkX, checkY) != player) break;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);

      EvaluationEntry evaluation =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (evaluation.counts.gomokuCount > 0) {
        return true;
      }
    }
  }
  return false;
}

// return 1: open three
// return 2: four
// return 3: gomoku
static int hasCapturableOnOpponentCriticalLine(Board* board, int x, int y, int dx, int dy,
                                               int player) {
  int checkX = x;
  int checkY = y;
  int opponent = OPPONENT(player);

  if (!(board->getValueBit(checkX + dx, checkY + dy) == opponent &&
        board->getValueBit(checkX + 2 * dx, checkY + 2 * dy) == opponent &&
        board->getValueBit(checkX + 3 * dx, checkY + 3 * dy) == player)) {
    dx = -dx;
    dy = -dy;
  }

  // forward
  for (int j = 0; j < 2; ++j) {
    checkX += dx;
    checkY += dy;
    if (board->getValueBit(checkX, checkY) != opponent) break;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);

      EvaluationEntry opponentEval =
          opponent == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (opponentEval.counts.openThreeCount) {
        return 1;
      }
      if (opponentEval.counts.openFourCount || opponentEval.counts.closedFourCount) {
        return 2;
      }
      if (opponentEval.counts.gomokuCount) {
        return 3;
      }
    }
  }

  return 0;
}

static bool hasCapturableOnPlayerVulnerableLine(Board* board, int x, int y, int dx, int dy,
                                                int player) {
  int checkX = x;
  int checkY = y;
  int opponent = OPPONENT(player);

  if (!(board->getValueBit(checkX + dx, checkY + dy) == opponent &&
        board->getValueBit(checkX + 2 * dx, checkY + 2 * dy) == opponent &&
        board->getValueBit(checkX + 3 * dx, checkY + 3 * dy) == player)) {
    dx = -dx;
    dy = -dy;
  }

  for (int j = 0; j < 2; ++j) {
    checkX += dx;
    checkY += dy;
    if (board->getValueBit(checkX, checkY) != opponent) break;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);

      EvaluationEntry opponentEval =
          opponent == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (opponentEval.counts.captureThreatCount) {
        return true;
      }
    }
  }

  return false;
}

static bool hasCapturableOnPlayerCriticalLine(Board* board, int x, int y, int dx, int dy,
                                              int player) {
  int checkX = x;
  int checkY = y;
  int opponent = OPPONENT(player);
  if (!(board->getValueBit(checkX + dx, checkY + dy) == opponent &&
        board->getValueBit(checkX + 2 * dx, checkY + 2 * dy) == opponent &&
        board->getValueBit(checkX + 3 * dx, checkY + 3 * dy) == player)) {
    dx = -dx;
    dy = -dy;
  }

  for (int j = 0; j < 2; ++j) {
    checkX += dx;
    checkY += dy;
    if (board->getValueBit(checkX, checkY) != opponent) break;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);
      EvaluationEntry playerEval =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (playerEval.counts.gomokuCount || playerEval.counts.openFourCount ||
          playerEval.counts.closedFourCount) {
        return true;
      }
    }
  }

  return false;
}

static bool hasCaptureBlockOnOpponentCriticalLine(Board* board, int x, int y, int dx, int dy,
                                                  int player) {
  int checkX = x;
  int checkY = y;
  int opponent = OPPONENT(player);
  if (!(board->getValueBit(checkX + dx, checkY + dy) == player &&
        board->getValueBit(checkX + 2 * dx, checkY + 2 * dy) == player &&
        board->getValueBit(checkX + 3 * dx, checkY + 3 * dy) == opponent)) {
    dx = -dx;
    dy = -dy;
  }

  for (int j = 0; j < 2; ++j) {
    checkX += dx;
    checkY += dy;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);
      EvaluationEntry opponentEval =
          opponent == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (opponentEval.counts.openFourCount || opponentEval.counts.gomokuCount) {
        return true;
      }
    }
  }

  return false;
}

static bool hasCapturableBreakingGomoku(Board* board, int x, int y, int dx, int dy, int player) {
  int checkX = x;
  int checkY = y;
  int opponent = OPPONENT(player);
  if (!(board->getValueBit(checkX + dx, checkY + dy) == opponent &&
        board->getValueBit(checkX + 2 * dx, checkY + 2 * dy) == opponent &&
        board->getValueBit(checkX + 3 * dx, checkY + 3 * dy) == player)) {
    dx = -dx;
    dy = -dy;
  }

  for (int j = 0; j < 2; ++j) {
    checkX += dx;
    checkY += dy;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);
      EvaluationEntry opponentEval =
          opponent == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      // Check if opponent is about to catch player stone
      int _checkX = checkX;
      int _checkY = checkY;
      int _checkDx = checkDx;
      int _checkDy = checkDy;

      if (opponentEval.counts.openTwoBlockCount) {
        // Check if capture vulnerable stone is on gomoku

        if (!(board->getValueBit(_checkX + checkDx, _checkY + checkDy) == opponent &&
              board->getValueBit(_checkX + 2 * checkDx, _checkY + 2 * checkDy) == player &&
              board->getValueBit(_checkX + 3 * checkDx, _checkY + 3 * checkDy) == player)) {
          _checkDx = -_checkDx;
          _checkDy = -_checkDy;
        }

        for (int jj = 0; jj < 2; ++jj) {
          _checkX += _checkDx;
          _checkY += _checkDy;

          for (int ii = 0; ii < 4; ++ii) {
            int _checkDx = DIRECTIONS[ii][0];
            int _checkDy = DIRECTIONS[ii][1];
            if ((_checkDx == checkDx && _checkDy == checkDy) ||
                (checkDx == -checkDx && checkDy == -checkDy))
              continue;
            unsigned int combined =
                extractLineAsBitsFromBoard(board, _checkX, _checkY, _checkDx, _checkDy);
            EvaluationEntry playerEval =
                player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

            if (playerEval.counts.gomokuCount) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}
// Check if it's non vulnerable line
static bool isNonVulnerableLine(Board* board, int x, int y, int dx, int dy, int player) {
  int checkX = x;
  int checkY = y;

  // if (board->getValueBit(checkX + dx, checkY + dy) != player) {
  //   dx = -dx;
  //   dy = -dy;
  // }

  for (int i = 0; i < 4; ++i) {
    int checkDx = DIRECTIONS[i][0];
    int checkDy = DIRECTIONS[i][1];
    if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

    // middle
    unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);
    EvaluationEntry playerEval =
        player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];
    if (playerEval.counts.captureVulnerable > 0) {
      return false;
    }

    // forward
    while (board->getValueBit(checkX + dx, checkY + dy) == player) {
      checkX += dx;
      checkY += dy;
      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);
      EvaluationEntry playerEval =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];
      if (playerEval.counts.captureVulnerable > 0) {
        return false;
      }
    }

    // backward
    checkX = x;
    checkY = y;
    while (board->getValueBit(checkX - dx, checkY - dy) == player) {
      checkX -= dx;
      checkY -= dy;
      unsigned int combined = extractLineAsBitsFromBoard(board, checkX, checkY, checkDx, checkDy);
      EvaluationEntry playerEval =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];
      if (playerEval.counts.captureVulnerable > 0) {
        return false;
      }
    }
  }

  return true;
}

int evaluatePositionHard(Board*& board, int player, int x, int y) {
  EvaluationEntry total;
  int activeCaptureScore = (player == board->getLastPlayer()) ? board->getLastPlayerScore()
                                                              : board->getNextPlayerScore();
  int opponentCaptureScore = (player == board->getLastPlayer()) ? board->getNextPlayerScore()
                                                                : board->getLastPlayerScore();

  // for checking player's & opponent's capture direction
  std::vector<int> captureDirections;
  std::vector<int> opponentCaptureDirections;
  std::vector<int> gomokuDirections;
  std::vector<int> openFourDirections;
  std::vector<int> captureBlockDirections;
  for (int i = 0; i < 4; ++i) {
    EvaluationEntry playerAxisScore =
        evaluateCombinedAxisHard(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);
    EvaluationEntry opponentAxisScore =
        evaluateCombinedAxisHard(board, OPPONENT(player), x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);
    // when capture occurs, store the direction
    if (playerAxisScore.counts.captureCount > 0) captureDirections.push_back(i);
    if (opponentAxisScore.counts.captureCount > 0) opponentCaptureDirections.push_back(i);
    if (playerAxisScore.counts.gomokuCount > 0) gomokuDirections.push_back(i);
    if (playerAxisScore.counts.openFourCount > 0) openFourDirections.push_back(i);
    if (playerAxisScore.counts.captureBlockCount > 0) captureBlockDirections.push_back(i);
    total += playerAxisScore;
  }

  /*
  ** Fine tuning for gomoku evaluation
  */

  // 1. HIGH PRIORITY CASE
  // - 1) If the player can reach the target score by capture, he must take it.
  if (activeCaptureScore + total.counts.captureCount >= board->getGoal()) {
    total.counts.captureWin += 1;
  }
  // - 2) If the player can't win by breakable gomoku, he must solve it.
  if (total.counts.closedThreeCount) {
    for (std::vector<int>::iterator it = opponentCaptureDirections.begin();
         it != opponentCaptureDirections.end(); ++it) {
      int dx = DIRECTIONS[*it][0];
      int dy = DIRECTIONS[*it][1];
      bool hasGomoku = hasGomokuOnClosedThree(board, x, y, dx, dy, player);
      if (hasGomoku) {
        total.counts.fixBreakableGomoku += 1;
      }
    }
  }

  // - 3) If the player can make perfect gomoku, he must take it
  if (total.counts.gomokuCount > 0) {
    for (std::vector<int>::iterator it = gomokuDirections.begin(); it != gomokuDirections.end();
         ++it) {
      int dx = DIRECTIONS[*it][0];
      int dy = DIRECTIONS[*it][1];
      bool isPerfect = isNonVulnerableLine(board, x, y, dx, dy, player);

      if (isPerfect) {
        total.counts.perfectCritical += 1;
      }
    }
  }
  if (total.counts.openFourCount > 0) {
    for (std::vector<int>::iterator it = openFourDirections.begin(); it != openFourDirections.end();
         ++it) {
      int dx = DIRECTIONS[*it][0];
      int dy = DIRECTIONS[*it][1];
      bool isPerfect = isNonVulnerableLine(board, x, y, dx, dy, player);

      if (isPerfect) {
        total.counts.perfectCritical += 1;
      }
    }
  }

  // 3. DEFENSE CASE

  // - 1) If player can break opponent's open 3+ or 4 stone, he must break.
  if (total.counts.captureCount > 0) {
    for (std::vector<int>::iterator it = captureDirections.begin(); it != captureDirections.end();
         ++it) {
      int dir = *it;
      int dx = DIRECTIONS[dir][0];
      int dy = DIRECTIONS[dir][1];
      // check if capturable spot is on opponent's critical line
      int hasCapturable = hasCapturableOnOpponentCriticalLine(board, x, y, dx, dy, player);
      if (hasCapturable) {
        total.counts.captureCriticalCount += 1;
        switch (hasCapturable) {
          case 1:
            total.counts.openThreeBlockCount += 1;
            break;
          case 2:
            total.counts.fourBlockCount += 1;
            break;
          case 3:
            total.counts.gomokuBlockCount += 1;
            break;
          default:
            break;
        }
      }

      // check if capturable spot can remove player's vulnerable spot
      hasCapturable = hasCapturableOnPlayerVulnerableLine(board, x, y, dx, dy, player);
      if (hasCapturable) {
        total.counts.captureCriticalCount += 1;
      }
      // check if capturable the opponent blocking player's critical line
      hasCapturable = hasCapturableOnPlayerCriticalLine(board, x, y, dx, dy, player);
      if (hasCapturable) {
        total.counts.captureCriticalCount += 1;
      }
      // check if capturable the opponent disturbing player's perfect gomoku
      hasCapturable = hasCapturableBreakingGomoku(board, x, y, dx, dy, player);
      if (hasCapturable) {
        total.counts.fixBreakableGomoku += 1;
      }
    }

    captureDirections.clear();
  }

  // - 2) If player can block capture on opponent's critical line, he must block
  if (total.counts.captureBlockCount > 0) {
    for (std::vector<int>::iterator it = captureBlockDirections.begin();
         it != captureBlockDirections.end(); ++it) {
      int dx = DIRECTIONS[*it][0];
      int dy = DIRECTIONS[*it][1];
      if (hasCaptureBlockOnOpponentCriticalLine(board, x, y, dx, dy, player)) {
        total.counts.captureBlockCriticalCount += 1;
      }
    }
  }

  // Score calculation
  // Critical Case
  if (total.counts.gomokuCount && !total.counts.perfectCritical) {
    // if gomoku is not perfect, block opponent opponent first...
    total.score += total.counts.gomokuCount * CONTINUOUS_OPEN_4;
  } else {
    total.score += total.counts.gomokuCount * GOMOKU;
  }
  total.score += total.counts.captureWin * CAPTURE_WIN;
  total.score += total.counts.fixBreakableGomoku * (GOMOKU * 2);
  total.score += total.counts.perfectCritical * PERFECT_CRITICAL_LINE;
  // Attack Case
  // - 1) If player can catch, he must catch.
  total.score += total.counts.captureCount * CAPTURE;
  // - 2) If player can threat, he must threat
  total.score += total.counts.threatCount * THREAT;
  total.score += total.counts.captureThreatCount * CAPTURE_THREAT;
  total.score += total.counts.openFourCount * CONTINUOUS_OPEN_4;
  total.score += total.counts.closedFourCount * CONTINUOUS_CLOSED_4;
  total.score += total.counts.openThreeCount * CONTINUOUS_OPEN_3;
  total.score += total.counts.closedThreeCount * CONTINUOUS_CLOSED_3;
  total.score += total.counts.openTwoCount * CONTINUOUS_OPEN_2;

  // Defense Case
  // - 1) Center priority
  int boardCenter = BOARD_SIZE / 2;
  total.score += CENTER_BONUS - (abs(x - boardCenter) + abs(y - boardCenter)) * 100;
  // - 2) Avoid capture vulnerability
  float vulnerablePenaltyCoefficient =
      std::max(float(opponentCaptureScore + 1) / (board->getGoal()), float(0.5));
  total.score -=
      total.counts.captureVulnerable * CAPTURE_VULNERABLE_PENALTY * vulnerablePenaltyCoefficient;
  // - 3) Avoid opponent double three spot (no priority)
  if (total.counts.openTwoBlockCount >= 2) {
    total.score -= total.counts.openTwoBlockCount * DOUBLE_THREE_PENALTY;
  }

  // - 4) Avoid capture
  total.score += total.counts.captureBlockCount * CAPTURE;
  // - 5) If opponent made open three or four, player must block it
  total.score += total.counts.fourBlockCount * BLOCK_CRITICAL_LINE;
  total.score += total.counts.openThreeBlockCount * BLOCK_CRITICAL_LINE;
  total.score += total.counts.captureCriticalCount * CAPTURE_CRITICAL;
  total.score += total.counts.captureBlockCriticalCount * CAPTURE_BLOCK_CRITICAL;
  total.score += total.counts.gomokuBlockCount * BLOCK_GOMOKU;
  // - 6) Simple block
  total.score += total.counts.openTwoBlockCount * BLOCK_OPEN_2;
  total.score += total.counts.openOneBlockCount * BLOCK_OPEN_1;
  total.score += total.counts.closedThreeBlockCount * BLOCK_CLOSE_3;

  // printEvalEntry(total);

  return total.score;
}

}  // namespace Evaluation
