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

  // if continuous on both side, it'll be gomoku (5 in a row)
  if (totalContinuous >= 4) {
    // returnValue.counts.openFourCount += 1;
    totalContinuous = 4;
  }

  // prevent from placing unnecessary moves by setting total continuous 0
  if (totalContinuous < 4) {
    // 1. for condition where total continous are less than equal to
    // if both ends are closed, it is meaningless to place the stone.
    if (backwardClosedEnd == true && forwardClosedEnd == true) {
      // std::cout << "check here " << std::endl;
      totalContinuous = 0;

    }

    // 2. if the total continuous + continuous empty => potential growth for gomoku is less then
    // five, don't need to extend the line
    else if (!((totalContinuous + forwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty + forwardContinuousEmpty) >= 5))
      totalContinuous = 0;

    // 3. prevent from opponent to capture (needs to check if necessary)
    // separated if condition because it needs to check all above then add
    // if (totalContinuous == 0 &&
    //     (isCaptureWarning(forward, player, false) || isCaptureWarning(backward, player, true)))
    //   totalContinuous = forwardContinuous + backwardContinuous;
  }

  if (isCaptureVulnerable(forward, backward, player)) {
    returnValue.counts.captureVulnerable += 1;
    totalContinuous = 0;
  }

  if (totalContinuous >= 4) totalContinuous = 4;

  if (totalContinuous == 3) {
    if (!forwardClosedEnd && !backwardClosedEnd) {
      returnValue.counts.openFourCount += 1;
      returnValue.counts.threatCount += 1;
    } else if (!forwardClosedEnd || !backwardClosedEnd) {
      returnValue.counts.threatCount += 1;
      returnValue.counts.closedFourCount += 1;
    }
  }

  if (totalContinuous == 2) {
    if (!forwardClosedEnd && !backwardClosedEnd) {
      // open three
      returnValue.counts.threatCount += 1;
      returnValue.counts.openThreeCount += 1;
    } else if (!forwardClosedEnd || !backwardClosedEnd) {
      // closed three
      returnValue.counts.threatCount += 1;
      returnValue.counts.closedThreeCount += 1;
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
  if (totalBlockCont >= 4) {
    returnValue.counts.immediateBlockCount += 1;
    totalBlockCont = 4;
  }

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

  if (totalBlockCont == 2) {
    if (!forwardBlockClosedEnd && !backwardBlockClosedEnd) {
      // open three
      returnValue.counts.defensiveBlockCount += 1;
      returnValue.counts.openThreeBlockCount += 1;
    } else if (!forwardClosedEnd || !backwardClosedEnd) {
      // closed three
      returnValue.counts.defensiveBlockCount += 1;
      returnValue.counts.closedThreeBlockCount += 1;
    }
  }

  if (!forwardBlockClosedEnd && forwardBlockContinuous == 2) {
    returnValue.counts.captureThreatCount += 1;
  }
  if (!backwardClosedEnd && backwardContinuous == 2) {
    returnValue.counts.captureThreatCount += 1;
  }

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

  int fcont = 0, bcont = 0, fempty = 0, bempty = 0;
  bool fclosed = false, bclosed = false;

  slideWindowContinuous(forward, opponent, false, fcont, fclosed, fempty);
  slideWindowContinuous(revBackward, opponent, true, bcont, bclosed, bempty);

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

static int evaluateCaptureDir(Board* board, int x, int y, int dx, int dy, int player) {
  int score = 0;

  // forward
  unsigned int fwdBits = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  if (checkCapture(fwdBits, player) > 0) {
    if (checkOpponentCaptureLineScore(board, x + dx, y + dy, OPPONENT(player)) > 0)
      score += CAPTURE_SCORE;
    if (checkOpponentCaptureLineScore(board, x + 2 * dx, y + 2 * dy, OPPONENT(player)) > 0)
      score += CAPTURE_SCORE;
  }

  // backward (reverse)
  unsigned int bwdBits = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  unsigned int revBits = reversePattern(bwdBits, SIDE_WINDOW_SIZE);
  if (checkCapture(revBits, player) > 0) {
    if (checkOpponentCaptureLineScore(board, x - dx, y - dy, OPPONENT(player)) > 0)
      score += CAPTURE_SCORE;
    if (checkOpponentCaptureLineScore(board, x - 2 * dx, y - 2 * dy, OPPONENT(player)) > 0)
      score += CAPTURE_SCORE;
  }

  return score;
}

int evaluatePositionHard(Board*& board, int player, int x, int y) {
  EvaluationEntry total;

  // for checking capture direction
  std::vector<int> captureDirections;
  for (int i = 0; i < 4; ++i) {
    EvaluationEntry axisScore =
        evaluateCombinedAxisHard(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);
    // when capture occurs, store the direction
    if (axisScore.counts.captureCount > 0) captureDirections.push_back(i);
    total += axisScore;
  }

  // check v pattern
  for (int i = 1; i < 8; i += 2) {
    total.score += checkVPattern(board, player, x, y, i);
  }

  if (total.counts.immediateBlockCount > 0) {
    return BLOCK_LINE_5;
  }

  // ---------------------continuous-----------------

  if (total.counts.openFourCount > 2) {
    total.score += FORCED_WIN_SEQUENCE;
    total.score += DOUBLE_OPEN_FOUR;
  }

  if (total.counts.openFourCount == 1) total.score = OPEN_FOUR;
  if (total.counts.closedFourCount == 1) total.score = CLOSED_FOUR;

  if (total.counts.openThreeCount == 1) total.score = OPEN_THREE;
  if (total.counts.closedThreeCount == 1) total.score = CLOSED_THREE;

  if ((total.counts.threatCount >= 2)) total.score = DOUBLE_THREAT;

  if (total.counts.threatCount >= 3) total.score = FORK;

  // if (total.counts.captureCount >= 2) total.score += SCORE_CHAIN_CAPTURE_SETUP;
  // ---------------------continuous-----------------

  // --------------- block -----------------
  if (total.counts.openFourBlockCount) total.score = BLOCK_LINE_4;

  // blocking openthree
  if (total.counts.openThreeBlockCount) total.score += OPEN_THREE_BLOCK;

  // double blocking
  if (total.counts.defensiveBlockCount >= 2) total.score += COUNTER_THREAT;

  // to prevent capture threatening
  // if score increases, even garbage position will be proiritized
  if (total.counts.captureThreatCount > 0) {
    total.score += CAPTURE_THREAT * total.counts.captureThreatCount;
  }
  // --------------- block -----------------

  // central bonus
  int boardCenter = BOARD_SIZE / 2;
  int posBonus = SCORE_POSITIONAL_ADVANTAGE - (abs(x - boardCenter) + abs(y - boardCenter)) * 1000;
  total.score += posBonus;

  int activeCaptureScore = (player == board->getLastPlayer()) ? board->getLastPlayerScore()
                                                              : board->getNextPlayerScore();
  int opponentCaptureScore = (player == board->getLastPlayer()) ? board->getNextPlayerScore()
                                                                : board->getLastPlayerScore();

  // for capture
  if (total.counts.captureCount > 0) {
    // if capture meets goal score, return score immediately
    if (activeCaptureScore + total.counts.captureCount >= board->getGoal()) return CAPTURE_WIN;
    // check if captureable spot opponent has bigger than OPEN_THREE
    // if bigger than OPEN_THREE, add CAPTURE_SCORE
    for (std::vector<int>::iterator it = captureDirections.begin(); it != captureDirections.end();
         ++it) {
      int dir = *it;
      int dx = DIRECTIONS[dir][0];
      int dy = DIRECTIONS[dir][1];

      total.score += evaluateCaptureDir(board, x, y, dx, dy, player);
    }
    captureDirections.clear();
    total.score += CAPTURE_SCORE * (activeCaptureScore + total.counts.captureCount);
  }

  // for block capture
  if (total.counts.captureBlockCount > 0) {
    if (opponentCaptureScore + total.counts.captureBlockCount >= board->getGoal())
      return BLOCK_LINE_5;
    total.score += CAPTURE_SCORE * (opponentCaptureScore + total.counts.captureBlockCount);
  }
  // totalScore += (activeCaptureScore - opponentCaptureScore) * SCORE_OPPORTUNISTIC_CAPTURE;

  // if capture vulnerable, reset score to 0 regardless of the previous accumulated scores.
  if (total.counts.captureVulnerable > 0) {
    total.score = 0;
  }

  return total.score;
}

}  // namespace Evaluation
