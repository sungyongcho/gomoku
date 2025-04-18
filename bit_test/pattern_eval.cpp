#include <iostream>
#include <string>

#define GOMOKU 10000000
#define CONTINUOUS_LINE_4 100000
#define CONTINUOUS_LINE_3 1000
#define CONTINUOUS_LINE_2 100
#define CONTINUOUS_LINE_1 10
#define BLOCK_LINE_5 999000
#define BLOCK_LINE_4 10000
#define BLOCK_LINE_3 100
#define BLOCK_LINE_2 10
#define BLOCK_LINE_1 1

#define SIDE_WINDOW_SIZE 4  // Number of 2-bit windows in an 8-bit number
#define EMPTY_SPACE 0
#define OUT_OF_BOUNDS 3

inline unsigned int pack_cells_4(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  return (a << 6) | (b << 4) | (c << 2) | d;
}

inline unsigned int pack_cells_3(unsigned int a, unsigned int b, unsigned int c) {
  return (a << 4) | (b << 2) | c;
}

inline unsigned int pack_cells_2(unsigned int a, unsigned int b) { return (a << 2) | b; }

inline unsigned int pack_cells_1(unsigned int a) { return a; }

static const int continuousScores[6] = {
    0, CONTINUOUS_LINE_1, CONTINUOUS_LINE_2, CONTINUOUS_LINE_3, CONTINUOUS_LINE_4, GOMOKU};
static const int blockScores[6] = {
    0, BLOCK_LINE_1, BLOCK_LINE_2, BLOCK_LINE_3, BLOCK_LINE_4, BLOCK_LINE_5};

// Slides a 2-bit window over the 8-bit representation of 'side'.
// For each window position, it prints the full binary string with the current 2-bit window in
// brackets, then prints the window's bits (and "matches" if it equals 'player'). If 'reverse' is
// true, the sliding is done in reverse order.
void slideWindowContinuous(int side, int player, bool reverse, int &continuous, bool &isClosedEnd,
                           int &continuousEmpty, int &emptyThenContinuous) {
  int opponent = player == 1 ? 2 : 1;
  (void)reverse;
  int player_count = 0;
  int closed = 0;
  int player_begin = 0;
  bool emptyPassed = false;

  if (!reverse) {
    for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
      int target_bit = ((side >> ((SIDE_WINDOW_SIZE - i - 1) * 2)) & 0x03);
      if (target_bit == player) {
        if (continuous == i) continuous++;
        if (player_begin == 0) player_begin = i + 1;
        if (emptyPassed && emptyThenContinuous == i - 1) emptyThenContinuous++;
        player_count++;
      } else if (target_bit == opponent || target_bit == OUT_OF_BOUNDS) {
        if (closed == 0) closed = i + 1;
      } else if (target_bit == EMPTY_SPACE && !emptyPassed) {
        emptyPassed = true;
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
  int block_begin = 0;
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

// capture vulnerable (for player 1)
// forward
// 1. .?12
// 2. .1?2
// backward
// 3. 21?.
// 4. 2?1.

bool isCaptureVulnerable(int forward, int backward, int player) {
  int opponent = player == 1 ? 2 : 1;

  if (((forward & 0xF0) >> 4) == pack_cells_2(player, opponent) &&
      (backward & 0x03 == EMPTY_SPACE)) {
    std::cout << "1" << std::endl;
    return true;
  }
  if (((forward & 0xC0) >> 6) == opponent &&
      (backward & 0x0F) == pack_cells_2(EMPTY_SPACE, player)) {
    std::cout << "2" << std::endl;
    return true;
  }
  if (((backward & 0x0F) == pack_cells_2(opponent, player)) &&
      (((forward & 0xC0) >> 6) == EMPTY_SPACE)) {
    std::cout << "3" << std::endl;
    return true;
  }

  if (((backward & 0x03) == opponent) &&
      (((forward & 0x0F) >> 4) == pack_cells_2(player, EMPTY_SPACE))) {
    std::cout << "4" << std::endl;
    return true;
  }

  return false;
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

int main() {
  int player = 1;
  int forward = 0b00010100;
  int backward = 0b00100101;

  int forwardContinuous = 0;
  bool forwardClosedEnd = false;
  int forwardContinuousEmpty = 0;
  int forwardEmptyThenContinuous = 0;

  int backwardContinuous = 0;
  bool backwardClosedEnd = false;
  int backwardContinuousEmpty = 0;
  int backwardEmptyThenContinuous = 0;

  int score = 0;

  printAxis(forward, backward);
  // Slide window in forward direction.
  slideWindowContinuous(forward, player, false, forwardContinuous, forwardClosedEnd,
                        forwardContinuousEmpty, forwardEmptyThenContinuous);
  slideWindowContinuous(backward, player, true, backwardContinuous, backwardClosedEnd,
                        backwardContinuousEmpty, backwardEmptyThenContinuous);

  std::cout << "----------------" << std::endl;
  std::cout << "forward continuous: " << forwardContinuous << std::endl;
  std::cout << "forward is closed end: " << forwardClosedEnd << std::endl;
  std::cout << "forward continuous empty: " << forwardContinuousEmpty << std::endl;
  std::cout << "forward empty then continuous: " << forwardEmptyThenContinuous << std::endl;
  std::cout << "----------------" << std::endl;

  std::cout << "----------------" << std::endl;
  std::cout << "backward continuous: " << backwardContinuous << std::endl;
  std::cout << "backward is closed end: " << backwardClosedEnd << std::endl;
  std::cout << "backward continuous empty: " << backwardContinuousEmpty << std::endl;
  std::cout << "backward empty then continuous: " << backwardEmptyThenContinuous << std::endl;
  std::cout << "----------------" << std::endl;

  int totalContinuous = forwardContinuous + backwardContinuous;

  // 1.if continous on both side, it'll be gomoku (5 in a row)
  if (totalContinuous >= 4) totalContinuous = 4;

  if (totalContinuous < 4) {
    // 2. for condition where total continous are less than equal to
    // if both ends are closed, it is meaningless to place the stone.
    if (backwardClosedEnd == true && forwardClosedEnd == true) {
      std::cout << "here1" << std::endl;
      totalContinuous = 0;
    }

    // 3. if the total continuous + continuous empty => potential growth for gomoku is less then
    // five, don't need to extend the line
    else if (!((totalContinuous + forwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty) >= 5 ||
               (totalContinuous + backwardContinuousEmpty + forwardContinuousEmpty) >= 5)) {
      std::cout << "here2 " << totalContinuous << std::endl;
      totalContinuous = 0;
    }

    // 4. prevent from opponent to capture (needs to check if necessary)
    // separated if condition because it needs to check all above then add
    if (isCaptureWarning(forward, player, false) || isCaptureWarning(backward, player, true)) {
      std::cout << "here3" << std::endl;
      totalContinuous = forwardContinuous + backwardContinuous;
    }
  }

  if (isCaptureVulnerable(forward, backward, player)) totalContinuous = 0;

  score += continuousScores[totalContinuous + 1];

  std::cout << "continuous: " << totalContinuous << " score: " << score << std::endl;
  std::cout << "=====================================" << std::endl;

  int forwardBlockContinuous = 0;
  bool forwardBlockClosedEnd = false;

  int backwardBlockContinuous = 0;
  bool backwardBlockClosedEnd = false;
  slideWindowBlock(forward, player, false, forwardBlockContinuous, forwardBlockClosedEnd);
  slideWindowBlock(backward, player, true, backwardBlockContinuous, backwardBlockClosedEnd);

  std::cout << "----------------" << std::endl;
  std::cout << "forward block continuous: " << forwardBlockContinuous << std::endl;
  std::cout << "forward block is closed end: " << forwardBlockClosedEnd << std::endl;
  std::cout << "----------------" << std::endl;

  std::cout << "----------------" << std::endl;
  std::cout << "backward continuous: " << backwardBlockContinuous << std::endl;
  std::cout << "backward block is closed end: " << backwardBlockClosedEnd << std::endl;
  std::cout << "----------------" << std::endl;
  std::cout << false << std::endl;

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

  std::cout << "block: " << totalBlockCont << " score: " << blockScores[totalBlockCont + 1]
            << std::endl;
  std::cout << "=====================================" << std::endl;

  return 0;
}
