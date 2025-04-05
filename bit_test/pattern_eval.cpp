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
void slideWindow(int side, int player, bool reverse, int &continuous, bool &isClosedEnd,
                 int &continuousEmpty) {
  int opponent = player == 1 ? 2 : 1;
  (void)reverse;
  int player_count = 0;
  int closed = 0;
  int player_begin = 0;

  if (!reverse) {
    for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
      int target_bit = ((side >> ((SIDE_WINDOW_SIZE - i - 1) * 2)) & 0x03);
      std::cout << target_bit << std::endl;
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
      std::cout << target_bit << std::endl;
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

  // if (player_count == continuous) {
  //   std::cout << "continuous in: " << continuous << std::endl;
  //   if (closed - continuous == 1) {
  //     std::cout << "but closed" << std::endl;
  //     isClosedEnd = true;
  //   }
  // }
  std::cout << "----------------" << std::endl;
}

bool isCaptureThreatning(int side, int player) {
  int opponent = player == 1 ? 2 : 1;

  if (((side & 0xFC) >> 2) == pack_cells_3(player, player, opponent)) return true;
  return false;
}

int main() {
  int forward = 0b01011000;
  int backward = 0b00100101;
  int player = 1;

  int forwardContinuous = 0;
  bool forwardClosedEnd = false;
  int forwardContinuousEmpty = 0;

  int backwardContinuous = 0;
  bool backwardClosedEnd = false;
  int backwardContinuousEmpty = 0;
  // Slide window in forward direction.
  slideWindow(forward, player, false, forwardContinuous, forwardClosedEnd, forwardContinuousEmpty);
  slideWindow(backward, player, true, backwardContinuous, backwardClosedEnd,
              backwardContinuousEmpty);

  std::cout << "----------------" << std::endl;
  std::cout << "forward continuous: " << forwardContinuous << std::endl;
  std::cout << "forward is closed end: " << forwardClosedEnd << std::endl;
  std::cout << "forward continuous empty: " << forwardContinuousEmpty << std::endl;
  std::cout << "----------------" << std::endl;

  std::cout << "----------------" << std::endl;
  std::cout << "backward continuous: " << backwardContinuous << std::endl;
  std::cout << "backward is closed end: " << backwardClosedEnd << std::endl;
  std::cout << "backward continuous empty: " << backwardContinuousEmpty << std::endl;
  std::cout << "----------------" << std::endl;

  int totalContinuous = forwardContinuous + backwardContinuous;

  // 1.if continous on both side, it'll be gomoku (5 in a row)
  if (totalContinuous >= 4) totalContinuous = 5;

  if (totalContinuous < 4) {
    // 2. for condition where total continous are less than equal to
    // if both ends are closed, it is meaningless to place the stone.
    if (backwardClosedEnd == true && forwardClosedEnd == true) totalContinuous = 0;
  }

  return 0;
}
