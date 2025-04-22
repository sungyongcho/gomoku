#include "Evaluation.hpp"

namespace Evaluation {

void printEvalEntry(EvaluationEntry eval) {
  std::cout << "=== EvalEntry ===" << std::endl;
  std::cout << "Score: " << eval.score << std::endl;
  std::cout << "openFourCount: " << eval.counts.openFourCount << std::endl;
  std::cout << "closedFourCount: " << eval.counts.closedFourCount << std::endl;
  std::cout << "openThreeCount: " << eval.counts.openThreeCount << std::endl;
  std::cout << "closedThreeCount: " << eval.counts.closedThreeCount << std::endl;
  std::cout << "threatCount: " << eval.counts.threatCount << std::endl;
  std::cout << "captureCount: " << eval.counts.captureCount << std::endl;
  std::cout << "defensiveBlockCount: " << eval.counts.defensiveBlockCount << std::endl;
  std::cout << "openFourBlockCount: " << eval.counts.openFourBlockCount << std::endl;
  std::cout << "closedFourBlockCount: " << eval.counts.closedFourBlockCount << std::endl;
  std::cout << "openThreeBlockCount: " << eval.counts.openThreeBlockCount << std::endl;
  std::cout << "closedThreeBlockCount: " << eval.counts.closedThreeBlockCount << std::endl;
  std::cout << "captureVulnerable: " << eval.counts.captureVulnerable << std::endl;
  std::cout << "captureBlockCount: " << eval.counts.captureBlockCount << std::endl;
  std::cout << "captureThreatCount: " << eval.counts.captureThreatCount << std::endl;
  std::cout << "captureCriticalCount: " << eval.counts.captureCriticalCount << std::endl;
  std::cout << "captureWin: " << eval.counts.captureWin << std::endl;
  std::cout << "gomokuCount: " << eval.counts.gomokuCount << std::endl;
  std::cout << "fixBreakableGomoku: " << eval.counts.fixBreakableGomoku << std::endl;
  std::cout << "perfectGomoku: " << eval.counts.perfectGomoku << std::endl;
  std::cout << "=================" << std::endl;
}

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
    } else if (!forwardClosedEnd || !backwardClosedEnd) {
      returnValue.counts.threatCount += 1;
      returnValue.counts.closedFourCount += 1;
    }
  }

  // [.]_OO_
  if (((forwardEmptyThenContinuous == 2) || (backwardEmptyThenContinuous == 2)) &&
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
  int forwardBlockContinuous = 0;
  bool forwardBlockClosedEnd = false;
  int forwardBlockEmptyThenContinuous = 0;

  int backwardBlockContinuous = 0;
  bool backwardBlockClosedEnd = false;
  int backwardBlockEmptyThenContinuous = 0;
  slideWindowBlock(forward, player, false, forwardBlockContinuous, forwardBlockClosedEnd,
                   forwardBlockEmptyThenContinuous);
  slideWindowBlock(backward, player, true, backwardBlockContinuous, backwardBlockClosedEnd,
                   backwardBlockEmptyThenContinuous);

  int totalBlockCont = forwardBlockContinuous + backwardBlockContinuous;

  // if continuous opponent is bigger or equal, should block asap
  if (totalBlockCont >= 4) {
    returnValue.counts.openFourBlockCount += 1;
    returnValue.counts.defensiveBlockCount += 1;
    totalBlockCont = 4;
  }

  if (totalBlockCont < 4) {
    // if both end is blocked by player and continuous is less then three, there is no need to block
    if (forwardBlockClosedEnd && backwardBlockClosedEnd) totalBlockCont = 0;

    // for each side, if one side continuous but that side is already closed,
    // it doesn't need to be blocked 'yet', so heuristics can go for better score moves.
    else if ((forwardBlockClosedEnd && (forwardBlockContinuous == totalBlockCont)) ||
             (backwardBlockClosedEnd && (backwardBlockContinuous == totalBlockCont))) {
      totalBlockCont = 0;
    }
  }

  if (totalBlockCont == 3) {
    if ((forwardBlockContinuous == 3 && !forwardBlockClosedEnd) ||
        (backwardBlockContinuous == 3 && !backwardBlockClosedEnd)) {
      returnValue.counts.openThreeBlockCount += 1;
      returnValue.counts.defensiveBlockCount += 1;
    } else if (!forwardBlockClosedEnd && !backwardBlockClosedEnd) {
      returnValue.counts.defensiveBlockCount += 1;
      returnValue.counts.openThreeBlockCount += 1;
    } else if (!forwardBlockClosedEnd || !backwardBlockClosedEnd) {
      returnValue.counts.defensiveBlockCount += 1;
      returnValue.counts.closedFourBlockCount += 1;
    }
  }

  // Evaluate possible threat captures (_XX_)
  if (!forwardBlockClosedEnd && forwardBlockContinuous == 2) {
    returnValue.counts.captureThreatCount += 1;
  }
  if (!backwardBlockClosedEnd && backwardBlockContinuous == 2) {
    returnValue.counts.captureThreatCount += 1;
  }

  // Check player captures (OXX_) cases
  if (checkCapture(forward, player) > 0) returnValue.counts.captureCount += 1;
  if (checkCapture(reversePattern(backward, SIDE_WINDOW_SIZE), player) > 0)
    returnValue.counts.captureCount += 1;

  // Check opponent capture (XOO_) cases
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

// void getForwardData(int forward, int player, std::vector<int>& coords, bool& isOpenEnd) {
//   int contiguous = 0;
//   // Scan from highest-order bits down: coordinate +1 is i=SIDE_WINDOW_SIZE-1.
//   for (int i = SIDE_WINDOW_SIZE - 1; i >= 0; --i) {
//     int cell = (forward >> (i * 2)) & 0x03;
//     int coord = SIDE_WINDOW_SIZE - i;  // For i=3, coord = 1; i=2, coord = 2, etc.
//     if (cell == player) {
//       coords.push_back(coord);
//       contiguous++;
//     } else {
//       break;
//     }
//   }
//   // Check the cell immediately after the contiguous group.
//   if (contiguous < SIDE_WINDOW_SIZE) {
//     int nextIndex = SIDE_WINDOW_SIZE - contiguous - 1;
//     int nextCell = (forward >> (nextIndex * 2)) & 0x03;
//     isOpenEnd = (nextCell == EMPTY_SPACE);
//   } else {
//     isOpenEnd = false;  // All cells filled implies a boundary.
//   }
// }

// // For backward: Bit cell at coordinate -1 is stored in the lowest-order 2 bits.
// void getBackwardData(int backward, int player, std::vector<int>& coords, bool& isOpenEnd) {
//   int contiguous = 0;
//   for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
//     int cell = (backward >> (i * 2)) & 0x03;
//     int coord = i + 1;  // i=0 => -1, i=1 => -2, etc.
//     if (cell == player) {
//       coords.push_back(coord);
//       contiguous++;
//     } else {
//       break;
//     }
//   }
//   // Check the cell immediately after the contiguous group.
//   if (contiguous < SIDE_WINDOW_SIZE) {
//     int nextIndex = contiguous;
//     int nextCell = (backward >> (nextIndex * 2)) & 0x03;
//     isOpenEnd = (nextCell == EMPTY_SPACE);
//   } else {
//     isOpenEnd = false;
//   }
//   // Reverse so that farthest (most negative) coordinate is printed first.
//   std::reverse(coords.begin(), coords.end());
// }

// std::vector<std::pair<int, int> > getNearbyPositions(
//     const std::vector<std::pair<int, int> >& stones) {
//   // Build a set of original positions for fast lookup.
//   std::set<std::pair<int, int> > original;
//   for (size_t i = 0; i < stones.size(); ++i) {
//     original.insert(stones[i]);
//   }

//   std::set<std::pair<int, int> > uniqueNeighbors;

//   // Iterate over each stone.
//   for (size_t i = 0; i < stones.size(); ++i) {
//     int x = stones[i].first;
//     int y = stones[i].second;

//     // Check all 8 directions.
//     for (int dx = -1; dx <= 1; ++dx) {
//       for (int dy = -1; dy <= 1; ++dy) {
//         // Skip the center itself.
//         if (dx == 0 && dy == 0) continue;
//         int nx = x + dx;
//         int ny = y + dy;
//         // Only add if candidate neighbor is within board boundaries.
//         if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
//           // Exclude if this position is already one of the original stones.
//           if (original.find(std::make_pair(nx, ny)) == original.end()) {
//             uniqueNeighbors.insert(std::make_pair(nx, ny));
//           }
//         }
//       }
//     }
//   }

//   // Convert the set back to a vector.
//   return std::vector<std::pair<int, int> >(uniqueNeighbors.begin(), uniqueNeighbors.end());
// }
// std::vector<std::pair<int, int> > getThresholdOpponentTotal(Board* board) {
//   int capture_player = board->getLastPlayer();
//   int capture_opponent = OPPONENT(capture_player);
//   int x = board->getLastX();
//   int y = board->getLastY();
//   std::vector<std::pair<int, int> > total;

//   for (int i = 0; i < 4; i++) {
//     int dx = DIRECTIONS[i][0];
//     int dy = DIRECTIONS[i][1];
//     std::vector<int> forwardCoords, backwardCoords;
//     bool forwardOpen, backwardOpen;

//     unsigned int forward = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
//     unsigned int backward = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
//     unsigned int revBackward = reversePattern(backward, SIDE_WINDOW_SIZE);
//     // unsigned int combined =
//     //     (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) | (0 << (2 * SIDE_WINDOW_SIZE)) |
//     forward;

//     // printPattern(forward, 4);
//     // printPattern(revBackward, 4);

//     getForwardData(forward, capture_player, forwardCoords, forwardOpen);
//     getBackwardData(revBackward, capture_player, backwardCoords, backwardOpen);

//     // std::cout << forwardCoords.size() + backwardCoords.size() << std::endl;

//     if (forwardOpen && backwardOpen && (forwardCoords.size() + backwardCoords.size() == 2)) {
//       for (unsigned int i = 0; i < forwardCoords.size(); i++) {
//         total.push_back(std::make_pair(x + dx * forwardCoords[i], y + dy * forwardCoords[i]));
//       }
//       for (unsigned int i = 0; i < backwardCoords.size(); i++) {
//         total.push_back(std::make_pair(x + -dx * backwardCoords[i], y + -dy *
//         backwardCoords[i]));
//       }
//     }

//     else if (!(forwardOpen && backwardOpen) &&
//              (forwardCoords.size() + backwardCoords.size() == 3)) {
//       for (unsigned int i = 0; i < forwardCoords.size(); i++) {
//         total.push_back(std::make_pair(x + dx * forwardCoords[i], y + dy * forwardCoords[i]));
//       }
//       for (unsigned int i = 0; i < backwardCoords.size(); i++) {
//         total.push_back(std::make_pair(x + -dx * backwardCoords[i], y + -dy *
//         backwardCoords[i]));
//       }
//     }

//     else if ((forwardOpen && backwardOpen) && (forwardCoords.size() + backwardCoords.size() ==
//     3)) {
//       for (unsigned int i = 0; i < forwardCoords.size(); i++) {
//         total.push_back(std::make_pair(x + dx * forwardCoords[i], y + dy * forwardCoords[i]));
//       }
//       for (unsigned int i = 0; i < backwardCoords.size(); i++) {
//         total.push_back(std::make_pair(x + -dx * backwardCoords[i], y + -dy *
//         backwardCoords[i]));
//       }
//     }
//   }

//   if (!total.empty()) {
//     total.push_back(std::make_pair(x, y));
//   }

//   std::vector<std::pair<int, int> > capturable;
//   for (unsigned int i = 0; i < total.size(); i++) {
//     for (int j = 0; j < 8; j++) {
//       int dx = DIRECTIONS[j][0];
//       int dy = DIRECTIONS[j][1];
//       unsigned int backward_empty = board->getValueBit(total[i].first - dx, total[i].second -
//       dy); if (backward_empty != EMPTY_SPACE) continue; unsigned int forward =
//       board->extractLineAsBits(total[i].first, total[i].second, dx, dy, 2); if (forward ==
//       pack_cells_2(capture_player, capture_opponent))
//         capturable.push_back(std::make_pair(total[i].first - dx, total[i].second - dy));
//     }
//   }

//   return capturable;
// }

// std::vector<std::pair<int, int> > getThresholdOpponentNearby(
//     Board* board, std::vector<std::pair<int, int> > total) {
//   std::vector<std::pair<int, int> > nearby = getNearbyPositions(total);

//   for (std::vector<std::pair<int, int> >::iterator it = nearby.begin(); it != nearby.end();) {
//     if (board->getValueBit(it->first, it->second) != EMPTY_SPACE)
//       it = nearby.erase(it);  // erase returns the next iterator
//     else
//       ++it;
//   }

//   return nearby;
// }

static int evaluateContinuousLineScore(Board* board, int x, int y, int dx, int dy, int opponent) {
  unsigned int forward = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  unsigned int backward = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  unsigned int revBackward = reversePattern(backward, SIDE_WINDOW_SIZE);

  int fcont = 0, bcont = 0, fempty = 0, bempty = 0, femptyThenCont = 0, bemptyThenCont = 0,
      femptyEmptyThenCont = 0, bemptyEmptyThenCont = 0;
  bool fclosed = false, bclosed = false;

  slideWindowContinuous(forward, opponent, false, fcont, fclosed, fempty, femptyThenCont,
                        femptyEmptyThenCont);
  slideWindowContinuous(revBackward, opponent, true, bcont, bclosed, bempty, bemptyThenCont,
                        bemptyEmptyThenCont);

  int total = fcont + bcont;
  if (total == 3) {
    if (!fclosed && !bclosed) return OPEN_FOUR;
    if (!fclosed || bclosed || fclosed || !bclosed) return CLOSED_FOUR;
  }
  if (total == 2 && !fclosed && !bclosed) {
    return OPEN_THREE;
  }
  return 0;
}

int checkOpponentCaptureLineScore(Board* board, int x, int y, int opponent) {
  int best = 0;
  for (int i = 0; i < 4; ++i) {
    int dx = DIRECTIONS[i][0];
    int dy = DIRECTIONS[i][1];
    best = std::max(best, evaluateContinuousLineScore(board, x, y, dx, dy, opponent));
  }
  return best;
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

      unsigned int forwardBits =
          board->extractLineAsBits(checkX, checkY, checkDx, checkDy, SIDE_WINDOW_SIZE);
      unsigned int backwardBits =
          board->extractLineAsBits(checkX, checkY, -checkDx, -checkDy, SIDE_WINDOW_SIZE);
      unsigned int reversedBackwardBits = reversePattern(backwardBits, SIDE_WINDOW_SIZE);
      unsigned int combined = (reversedBackwardBits << (2 * (SIDE_WINDOW_SIZE + 1))) |
                              (0 << (2 * SIDE_WINDOW_SIZE)) | forwardBits;

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

      unsigned int forwardBits =
          board->extractLineAsBits(checkX, checkY, checkDx, checkDy, SIDE_WINDOW_SIZE);
      unsigned int backwardBits =
          board->extractLineAsBits(checkX, checkY, -checkDx, -checkDy, SIDE_WINDOW_SIZE);
      unsigned int reversedBackwardBits = reversePattern(backwardBits, SIDE_WINDOW_SIZE);
      unsigned int combined = (reversedBackwardBits << (2 * (SIDE_WINDOW_SIZE + 1))) |
                              (0 << (2 * SIDE_WINDOW_SIZE)) | forwardBits;

      EvaluationEntry evaluation =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (evaluation.counts.gomokuCount > 0) {
        return true;
      }
    }
  }
  return false;
}

static bool hasCapturableOnOpponentCriticalLine(Board* board, int x, int y, int dx, int dy,
                                                int player) {
  int checkX = x;
  int checkY = y;
  int opponent = OPPONENT(player);

  // forward
  for (int j = 0; j < 2; ++j) {
    checkX += dx;
    checkY += dy;
    if (board->getValueBit(checkX, checkY) != opponent) break;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int forwardBits =
          board->extractLineAsBits(checkX, checkY, checkDx, checkDy, SIDE_WINDOW_SIZE);
      unsigned int backwardBits =
          board->extractLineAsBits(checkX, checkY, -checkDx, -checkDy, SIDE_WINDOW_SIZE);
      unsigned int reversedBackwardBits = reversePattern(backwardBits, SIDE_WINDOW_SIZE);
      unsigned int combined = (reversedBackwardBits << (2 * (SIDE_WINDOW_SIZE + 1))) |
                              (0 << (2 * SIDE_WINDOW_SIZE)) | forwardBits;

      EvaluationEntry opponentEval =
          opponent == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (opponentEval.counts.openFourCount || opponentEval.counts.openThreeCount ||
          opponentEval.counts.closedFourCount || opponentEval.score >= GOMOKU) {
        return true;
      }
    }
  }

  // backward
  checkX = x;
  checkY = y;
  for (int j = 0; j < 2; ++j) {
    checkX -= dx;
    checkY -= dy;
    if (board->getValueBit(checkX, checkY) != opponent) break;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int forwardBits =
          board->extractLineAsBits(checkX, checkY, checkDx, checkDy, SIDE_WINDOW_SIZE);
      unsigned int backwardBits =
          board->extractLineAsBits(checkX, checkY, -checkDx, -checkDy, SIDE_WINDOW_SIZE);
      unsigned int reversedBackwardBits = reversePattern(backwardBits, SIDE_WINDOW_SIZE);
      unsigned int combined = (reversedBackwardBits << (2 * (SIDE_WINDOW_SIZE + 1))) |
                              (0 << (2 * SIDE_WINDOW_SIZE)) | forwardBits;

      EvaluationEntry opponentEval =
          opponent == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (opponentEval.counts.openFourCount || opponentEval.counts.openThreeCount ||
          opponentEval.counts.closedFourCount || opponentEval.score >= GOMOKU) {
        return true;
      }
    }
  }
  return false;
}

static bool hasPerfectGomoku(Board* board, int x, int y, int dx, int dy, int player) {
  int checkX = x;
  int checkY = y;
  int _dx = dx;
  int _dy = dy;

  // forward
  for (int j = 0; j < 2; ++j) {
    checkX += _dx;
    checkY += _dy;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int forwardBits =
          board->extractLineAsBits(checkX, checkY, checkDx, checkDy, SIDE_WINDOW_SIZE);
      unsigned int backwardBits =
          board->extractLineAsBits(checkX, checkY, -checkDx, -checkDy, SIDE_WINDOW_SIZE);
      unsigned int reversedBackwardBits = reversePattern(backwardBits, SIDE_WINDOW_SIZE);
      unsigned int combined = (reversedBackwardBits << (2 * (SIDE_WINDOW_SIZE + 1))) |
                              (0 << (2 * SIDE_WINDOW_SIZE)) | forwardBits;

      EvaluationEntry playerEval =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (playerEval.counts.captureVulnerable > 0) {
        return false;
      }
    }
  }

  // backward
  checkX = x;
  checkY = y;
  _dx = -dx;
  _dy = -dy;
  for (int j = 0; j < 2; ++j) {
    checkX += _dx;
    checkY += _dy;

    for (int i = 0; i < 4; ++i) {
      int checkDx = DIRECTIONS[i][0];
      int checkDy = DIRECTIONS[i][1];
      if ((checkDx == dx && checkDy == dy) || (checkDx == -dx && checkDy == -dy)) continue;

      unsigned int forwardBits =
          board->extractLineAsBits(checkX, checkY, checkDx, checkDy, SIDE_WINDOW_SIZE);
      unsigned int backwardBits =
          board->extractLineAsBits(checkX, checkY, -checkDx, -checkDy, SIDE_WINDOW_SIZE);
      unsigned int reversedBackwardBits = reversePattern(backwardBits, SIDE_WINDOW_SIZE);
      unsigned int combined = (reversedBackwardBits << (2 * (SIDE_WINDOW_SIZE + 1))) |
                              (0 << (2 * SIDE_WINDOW_SIZE)) | forwardBits;

      EvaluationEntry playerEval =
          player == PLAYER_1 ? patternPlayerOne[combined] : patternPlayerTwo[combined];

      if (playerEval.counts.captureCount > 0) {
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
  // int opponentCaptureScore = (player == board->getLastPlayer()) ? board->getNextPlayerScore()
  //                                                               : board->getLastPlayerScore();

  // for checking player's & opponent's capture direction
  std::vector<int> captureDirections;
  std::vector<int> opponentCaptureDirections;
  std::vector<int> gomokuDirections;
  for (int i = 0; i < 4; ++i) {
    EvaluationEntry playerAxisScore =
        evaluateCombinedAxisHard(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);
    EvaluationEntry opponentAxisScore =
        evaluateCombinedAxisHard(board, OPPONENT(player), x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);
    // when capture occurs, store the direction
    if (playerAxisScore.counts.captureCount > 0) captureDirections.push_back(i);
    if (opponentAxisScore.counts.captureCount > 0) opponentCaptureDirections.push_back(i);
    if (playerAxisScore.counts.openFourCount > 0 || playerAxisScore.counts.closedFourCount > 0)
      gomokuDirections.push_back(i);
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
      bool isPerfectGomoku = hasPerfectGomoku(board, x, y, dx, dy, player);

      if (isPerfectGomoku) {
        total.counts.perfectGomoku += 1;
      }
    }
  }

  // 3. DEFENSE CASE

  // - 5) If player can break opponent's open 3+ or 4 stone, he must break.
  if (total.counts.captureCount > 0) {
    // check if capturable spot opponent has bigger than OPEN_THREE
    // if bigger than OPEN_THREE, add CAPTURE_SCORE
    for (std::vector<int>::iterator it = captureDirections.begin(); it != captureDirections.end();
         ++it) {
      int dir = *it;
      int dx = DIRECTIONS[dir][0];
      int dy = DIRECTIONS[dir][1];
      bool hasCapturable = hasCapturableOnOpponentCriticalLine(board, x, y, dx, dy, player);
      if (hasCapturable) {
        total.counts.captureCriticalCount += 1;
      }
    }
    captureDirections.clear();
  }

  // Score calculation
  // Critical Case
  total.score += total.counts.gomokuCount * GOMOKU;
  total.score += total.counts.captureWin * CAPTURE_WIN;
  total.score += total.counts.fixBreakableGomoku * (GOMOKU * 2);
  total.score += total.counts.perfectGomoku * PERFECT_GOMOKU;
  // Attack Case
  // - 1) If player can catch, he must catch.
  total.score += total.counts.captureCount * CAPTURE;
  // - 2) If player can threat, he must threat
  total.score += total.counts.threatCount * THREAT;
  total.score += total.counts.captureThreatCount * THREAT;
  // Defense Case
  // - 1) Center priority
  int boardCenter = BOARD_SIZE / 2;
  total.score += CENTER_BONUS - (abs(x - boardCenter) + abs(y - boardCenter)) * 1000;
  // - 2) Avoid capture vulnerability
  total.score -= total.counts.captureVulnerable * CAPTURE_VULNERABLE_PENALTY;
  // - 3) Avoid capture
  total.score += total.counts.captureBlockCount * CAPTURE;
  // - 4) If opponent made open three or four, player must block it
  total.score += total.counts.openThreeBlockCount * THREAT_BLOCK;
  total.score += total.counts.openFourBlockCount * THREAT_BLOCK;
  total.score += total.counts.captureCriticalCount * CAPTURE_CRITICAL;

  // printEvalEntry(total);

  return total.score;
}

}  // namespace Evaluation
