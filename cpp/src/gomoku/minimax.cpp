#include "minimax.hpp"

#include <cstdlib>
#include <limits>
#include <sstream>

namespace Minimax {
int combinedPatternScoreTable[LOOKUP_TABLE_SIZE] = {0};

// (TODO) needs to be improved, this only checks the basic continous pattern
int evaluateCombinedPattern(int combinedPattern, int player) {
  int cells[COMBINED_WINDOW_SIZE];
  // Decode each cell (2 bits per cell).
  for (int i = 0; i < COMBINED_WINDOW_SIZE; i++) {
    int shift = 2 * (COMBINED_WINDOW_SIZE - 1 - i);
    cells[i] = (combinedPattern >> shift) & 0x3;
  }
  // The center index is SIDE_WINDOW_SIZE.
  int center = SIDE_WINDOW_SIZE;
  // Ensure the center cell is set to player's stone.
  cells[center] = player;

  // Count contiguous stones including the center.
  int leftCount = 0;
  for (int i = center - 1; i >= 0; i--) {
    if (cells[i] == player)
      leftCount++;
    else
      break;
  }
  int rightCount = 0;
  for (int i = center + 1; i < COMBINED_WINDOW_SIZE; i++) {
    if (cells[i] == player)
      rightCount++;
    else
      break;
  }
  int totalRun = leftCount + 1 + rightCount;

  // Check open ends: if the cell immediately outside the contiguous run is
  // EMPTY.
  bool openLeft = (center - leftCount - 1 >= 0 && cells[center - leftCount - 1] == EMPTY_SPACE);
  bool openRight = (center + rightCount + 1 < COMBINED_WINDOW_SIZE &&
                    cells[center + rightCount + 1] == EMPTY_SPACE);

  int score = 0;
  if (totalRun >= 5)
    score = GOMOKU;
  else if (totalRun == 4)
    score = (openLeft && openRight) ? OPEN_LINE_4 : BLOCKED_LINE_4;
  else if (totalRun == 3)
    score = (openLeft && openRight) ? OPEN_LINE_3 : BLOCKED_LINE_3;
  else if (totalRun == 2)
    score = (openLeft && openRight) ? OPEN_LINE_2 : BLOCKED_LINE_2;
  else
    score = (openLeft && openRight) ? OPEN_SINGLE_STONE : 0;

  // Check capture opportunities.
  int opponent = (player == PLAYER_1 ? PLAYER_2 : PLAYER_1);
  // Forward capture check: if cells at indices center+1, center+2, center+3
  // equal: opponent, opponent, player.
  if (center + 3 < COMBINED_WINDOW_SIZE) {
    if (cells[center + 1] == opponent && cells[center + 2] == opponent &&
        cells[center + 3] == player) {
      score += CAPTURE_SCORE;
    }
  }
  // Backward capture check: if cells at indices center-3, center-2, center-1
  // equal: player, opponent, opponent.
  if (center - 3 >= 0) {
    if (cells[center - 1] == opponent && cells[center - 2] == opponent &&
        cells[center - 3] == player) {
      score += CAPTURE_SCORE;
    }
  }
  return score;
}

void initCombinedPatternScoreTable() {
  for (int pattern = 0; pattern < LOOKUP_TABLE_SIZE; pattern++) {
    // Here we assume evaluation for PLAYER_1.
    // (For two-player support, either build two tables or adjust at runtime.)
    combinedPatternScoreTable[pattern] = evaluateCombinedPattern(pattern, PLAYER_1);
  }
}

inline unsigned int reversePattern(unsigned int pattern, int windowSize) {
  unsigned int reversed = 0;
  for (int i = 0; i < windowSize; i++) {
    reversed = (reversed << 2) | (pattern & 0x3);
    pattern >>= 2;
  }
  return reversed;
}

int evaluateCombinedAxis(Board *board, int player, int x, int y, int dx, int dy) {
  // Extract the forward window.
  unsigned int forward = board->extractLineAsBits(x, y, dx, dy, SIDE_WINDOW_SIZE);
  // Extract the backward window.
  unsigned int backward = board->extractLineAsBits(x, y, -dx, -dy, SIDE_WINDOW_SIZE);
  // Reverse the backward window so that the cell immediately adjacent to (x,y)
  // is at the rightmost position.
  unsigned int revBackward = reversePattern(backward, SIDE_WINDOW_SIZE);
  // Combine: [reversed backward window] + [center cell (player)] + [forward
  // window]
  unsigned int combined = (revBackward << (2 * (SIDE_WINDOW_SIZE + 1))) |
                          ((unsigned int)player << (2 * SIDE_WINDOW_SIZE)) | forward;
  int score = combinedPatternScoreTable[combined];
  return score;
}

int evaluatePosition(Board *&board, int player, int x, int y) {
  int totalScore = 0;

  if (board->getValueBit(x, y) == EMPTY_SPACE) return 0;

  for (int i = 0; i < 4; ++i)
    totalScore += evaluateCombinedAxis(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);

  return totalScore;
}
}  // namespace Minimax
