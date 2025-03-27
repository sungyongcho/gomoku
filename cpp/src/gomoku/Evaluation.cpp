#include "Evaluation.hpp"
// to remove
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

/*
 * 1. the evaluated cell from combined pattern will always be in the middle
 * 2. check leftside, rightside for the continous pattern
 * 2.1 for each side, check if the continous pattern happens in both on current player and opponent,
 *     the opponent score is for to check how dangerous the current position is.
 * 2.2 when checking for leftside and rightside, combine the score by checking the player's score
 *     first then subtract opponent's score by last.
 * 2.3 if the capture is available for player, give advantage and if for opponent, subtract for
 *     opponent.
 * ps1. assuming middle is always empty.
 * ps2. there are three patterns 0 for empty, 1 for p1, 2 for p2, 3 for out of bounds.
 */
int evaluateContinousPattern(unsigned int backward, unsigned int forward, unsigned int player) {
  int score = 0;
  unsigned int opponent = OPPONENT(player);

  if (forward == pack_cells_4(player, player, player, player))
    score += OPEN_LINE_4;
  else if (((forward & 0xFC) >> 2) == pack_cells_3(player, player, player))
    score += OPEN_LINE_3;
  else if (((forward & 0xF0) >> 4) == pack_cells_2(player, player))
    score += OPEN_LINE_2;
  else if (((forward & 0xC0) >> 6) == player)
    score += OPEN_LINE_1;
  else if (forward == pack_cells_4(opponent, opponent, opponent, opponent))
    score -= BLOCKED_LINE_4;
  else if (((forward & 0xFC) >> 2) == pack_cells_3(opponent, opponent, opponent))
    score -= BLOCKED_LINE_3;
  else if (((forward & 0xF0) >> 4) == pack_cells_2(opponent, opponent))
    score -= BLOCKED_LINE_2;
  else if (((forward & 0xC0) >> 6) == opponent)
    score -= BLOCKED_LINE_1;

  if (backward == pack_cells_4(player, player, player, player))
    score += OPEN_LINE_4;
  else if ((backward & 0x3F) == pack_cells_3(player, player, player))
    score += OPEN_LINE_3;
  else if ((backward & 0x0F) == pack_cells_2(player, player))
    score += OPEN_LINE_2;
  else if ((backward & 0x03) == player)
    score += OPEN_LINE_1;
  else if (backward == pack_cells_4(opponent, opponent, opponent, opponent))
    score -= BLOCKED_LINE_4;
  else if ((backward & 0x3F) == pack_cells_3(opponent, opponent, opponent))
    score -= BLOCKED_LINE_3;
  else if ((backward & 0x0F) == pack_cells_2(opponent, opponent))
    score -= BLOCKED_LINE_2;
  else if ((backward & 0x03) == opponent)
    score -= BLOCKED_LINE_1;

  return score;
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
      patternScoreTablePlayerOne[pattern] = evaluateContinousPattern(backward, forward, PLAYER_1);
      patternScoreTablePlayerTwo[pattern] = evaluateContinousPattern(backward, forward, PLAYER_2);
    }
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

int checkCapture(unsigned int side, unsigned int player) {
  unsigned int opponent = OPPONENT(player);
  if (((side & 0xFC) >> 2) == pack_cells_3(opponent, opponent, player))
    return CAPTURE_SCORE;
  else if (((side & 0xfc) >> 2) == pack_cells_3(player, player, opponent))
    return -CAPTURE_SCORE;
  return 0;
}

int evaluateCombinedAxis(Board *board, int player, int x, int y, int dx, int dy) {
  int score;
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
  int activeScore = (player == board->getLastPlayer()) ? board->getLastPlayerScore()
                                                       : board->getNextPlayerScore();
  int opponentScore = (player == board->getLastPlayer()) ? board->getNextPlayerScore()
                                                         : board->getLastPlayerScore();
  std::cout << "player: " << player << std::endl;
  std::cout << "nextPlayer: " << board->getNextPlayer() << " LastPlayer: " << board->getLastPlayer()
            << std::endl;
  std::cout << "activeScore: " << activeScore << " opponentScore: " << opponentScore << std::endl;
  std::cout << "forward" << std::endl;
  printPattern(forward, 4);
  printPattern(backward, 4);
  if (checkCapture(forward, player) > 0 || checkCapture(backward, player) > 0) {
    if (activeScore == 4) return GOMOKU;
    score += CAPTURE_SCORE * (activeScore + 1);
  } else if (checkCapture(forward, player) < 0 || checkCapture(backward, player) < 0) {
    if (opponentScore == 4) return OPEN_LINE_4 - 1;
    score -= CAPTURE_SCORE * (opponentScore + 1);
  }

  return score;
}

int evaluatePosition(Board *&board, int player, int x, int y) {
  (void)board;
  (void)player;
  (void)x;
  (void)y;
  int totalScore = 0;

  // if (board->getValueBit(x, y) == EMPTY_SPACE) return 0;

  for (int i = 0; i < 4; ++i)
    totalScore += evaluateCombinedAxis(board, player, x, y, DIRECTIONS[i][0], DIRECTIONS[i][1]);

  return totalScore;
}

}  // namespace Evaluation
