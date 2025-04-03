#include <iostream>
using namespace std;

#define EMPTY_SPACE 0
#define GOMOKU 10000
#define SIDE_WINDOW_SIZE 4

// Returns the opponent of the given player.
#define OPPONENT(p) ((p) == 1 ? 2 : 1)

inline unsigned int pack_cells_4(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  return (a << 6) | (b << 4) | (c << 2) | d;
}

inline unsigned int pack_cells_3(unsigned int a, unsigned int b, unsigned int c) {
  return (a << 4) | (b << 2) | c;
}

inline unsigned int pack_cells_2(unsigned int a, unsigned int b) { return (a << 2) | b; }

inline unsigned int pack_cells_1(unsigned int a) { return a; }

// Sample scoring arrays (you can adjust these as needed)
int continuousScores[5] = {0, 1, 10, 100, 1000};
int blockScores[5] = {0, 1, 5, 50, 500};

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

int evaluateContinousPattern(unsigned int backward, unsigned int forward, unsigned int player) {
  unsigned int opponent = OPPONENT(player);
  int continuous = 0;
  int forwardContinuous = 0;
  int forwardContEmpty = 0;
  int backwardContinuous = 0;
  int backwardContEmpty = 0;
  int block = 0;

  if (forward == pack_cells_4(player, player, player, player)) {
    cout << "here4" << endl;
    return GOMOKU;
  } else if (((forward & 0xFC) >> 2) == pack_cells_3(player, player, player)) {
    cout << "here3" << endl;
    forwardContinuous += 3;
  } else if (((forward & 0xF0) >> 4) == pack_cells_2(player, player)) {
    cout << "here2" << endl;
    forwardContinuous += 2;
  } else if (((forward & 0xC0) >> 6) == player) {
    cout << "here1" << endl;
    forwardContinuous += 1;
  } else if (forward == pack_cells_4(opponent, opponent, opponent, opponent))
    block += 4;
  else if (((forward & 0xFC) >> 2) == pack_cells_3(opponent, opponent, opponent))
    block += 3;
  else if (((forward & 0xF0) >> 4) == pack_cells_2(opponent, opponent))
    block += 2;
  else if (((forward & 0xC0) >> 6) == opponent)
    block += 1;

  // Extend forward continuous pattern into empty spaces.
  for (int i = 4 - forwardContinuous; i > 0; i--) {
    if (((forward >> ((i - 1) * 2)) & 0x03) == EMPTY_SPACE)
      forwardContEmpty += 1;
    else
      break;
  }

  if (backward == pack_cells_4(player, player, player, player)) {
    return GOMOKU;
  } else if ((backward & 0x3F) == pack_cells_3(player, player, player)) {
    backwardContinuous += 3;
  } else if ((backward & 0x0F) == pack_cells_2(player, player)) {
    backwardContinuous += 2;
  } else if ((backward & 0x03) == player) {
    backwardContinuous += 1;
  } else if (backward == pack_cells_4(opponent, opponent, opponent, opponent))
    block += 4;
  else if ((backward & 0x3F) == pack_cells_3(opponent, opponent, opponent))
    block += 3;
  else if ((backward & 0x0F) == pack_cells_2(opponent, opponent))
    block += 2;
  else if ((backward & 0x03) == opponent)
    block += 1;

  // Count empty spaces in the backward direction.
  for (int i = backwardContinuous; i < SIDE_WINDOW_SIZE; i++) {
    if (((backward >> (i * 2)) & 0x03) == EMPTY_SPACE)
      backwardContEmpty += 1;
    else
      break;
  }
  printPattern(forward, 4);
  printPattern(backward, 4);
  cout << "forward continous: " << forwardContinuous << " " << forwardContEmpty << endl;
  cout << "backward continous: " << backwardContinuous << " " << backwardContEmpty << endl;

  continuous = forwardContinuous + backwardContinuous;
  if (continuous > 4) continuous = 4;

  cout << continuous << endl;
  if ((SIDE_WINDOW_SIZE - continuous) > (forwardContEmpty + backwardContEmpty)) continuous = 0;
  if (block > 4) block = 4;

  return continuousScores[continuous] + blockScores[block];
}

int main() {
  unsigned int player = 1;  // Assume player 1 for testing
  unsigned int opponent = OPPONENT(player);
  int score;
  int score_opponent;

  // Test case 1: Forward is four in a row for player -> should return GOMOKU
  unsigned int forward = pack_cells_4(opponent, EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE);
  unsigned int backward = pack_cells_4(EMPTY_SPACE, player, opponent, opponent);
  // score = evaluateContinousPattern(backward, forward, player);
  score_opponent = evaluateContinousPattern(backward, forward, opponent);

  cout << "Test case 1 (GOMOKU): score = " << 0 << " " << score_opponent << endl;

  return 0;
}
