#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

#define SIDE_WINDOW_SIZE 4
#define EMPTY_SPACE 0
#define PLAYER 1
#define OPPONENT 2
#define OUT_OF_BOUNDS 3

// Prints the backward and forward bit representations in board order.
void printAxis(int forward, int backward) {
  // Process backward 8 bits (from MSB to LSB) to print left side.
  for (int i = SIDE_WINDOW_SIZE - 1; i >= 0; --i) {
    int val = (backward >> (i * 2)) & 0x03;
    switch (val) {
      case EMPTY_SPACE:
        cout << ".";
        break;
      case PLAYER:
        cout << "1";
        break;
      case OPPONENT:
        cout << "2";
        break;
      case OUT_OF_BOUNDS:
        cout << "X";
        break;
    }
  }
  // Print center marker.
  cout << "[player = 1]";
  // Process forward 8 bits (from MSB to LSB) to print right side.
  for (int i = SIDE_WINDOW_SIZE - 1; i >= 0; --i) {
    int val = (forward >> (i * 2)) & 0x03;
    switch (val) {
      case EMPTY_SPACE:
        cout << ".";
        break;
      case PLAYER:
        cout << "1";
        break;
      case OPPONENT:
        cout << "2";
        break;
      case OUT_OF_BOUNDS:
        cout << "X";
        break;
    }
  }
  cout << endl;
}

// For forward: Bit cell at coordinate +1 is stored in the highest-order 2 bits.
void getForwardData(int forward, int player, vector<int>& coords, bool& isOpenEnd) {
  int contiguous = 0;
  // Scan from highest-order bits down: coordinate +1 is i=SIDE_WINDOW_SIZE-1.
  for (int i = SIDE_WINDOW_SIZE - 1; i >= 0; --i) {
    int cell = (forward >> (i * 2)) & 0x03;
    int coord = SIDE_WINDOW_SIZE - i;  // For i=3, coord = 1; i=2, coord = 2, etc.
    if (cell == player) {
      coords.push_back(coord);
      contiguous++;
    } else {
      break;
    }
  }
  // Check the cell immediately after the contiguous group.
  if (contiguous < SIDE_WINDOW_SIZE) {
    int nextIndex = SIDE_WINDOW_SIZE - contiguous - 1;
    int nextCell = (forward >> (nextIndex * 2)) & 0x03;
    isOpenEnd = (nextCell == EMPTY_SPACE);
  } else {
    isOpenEnd = false;  // All cells filled implies a boundary.
  }
}

// For backward: Bit cell at coordinate -1 is stored in the lowest-order 2 bits.
void getBackwardData(int backward, int player, vector<int>& coords, bool& isOpenEnd) {
  int contiguous = 0;
  for (int i = 0; i < SIDE_WINDOW_SIZE; ++i) {
    int cell = (backward >> (i * 2)) & 0x03;
    int coord = -(i + 1);  // i=0 => -1, i=1 => -2, etc.
    if (cell == player) {
      coords.push_back(coord);
      contiguous++;
    } else {
      break;
    }
  }
  // Check the cell immediately after the contiguous group.
  if (contiguous < SIDE_WINDOW_SIZE) {
    int nextIndex = contiguous;
    int nextCell = (backward >> (nextIndex * 2)) & 0x03;
    isOpenEnd = (nextCell == EMPTY_SPACE);
  } else {
    isOpenEnd = false;
  }
  // Reverse so that farthest (most negative) coordinate is printed first.
  reverse(coords.begin(), coords.end());
}

int main() {
  int player = PLAYER;
  // Example bit patterns:
  // backward = 0b00100101 prints as ".211"
  // forward  = 0b01010100 prints as "111."
  int backward = 0b00100010;
  int forward = 0b01010100;

  // Print the board row.
  cout << "Board row:" << endl;
  printAxis(forward, backward);

  // Get forward and backward contiguous stone coordinates and open/closed flags.
  vector<int> forwardCoords, backwardCoords;
  bool forwardOpen, backwardOpen;

  getForwardData(forward, player, forwardCoords, forwardOpen);
  getBackwardData(backward, player, backwardCoords, backwardOpen);

  // Print backward coordinates with open/closed status.
  cout << "Backward side: ";
  for (size_t i = 0; i < backwardCoords.size(); ++i) cout << "(" << backwardCoords[i] << ") ";
  cout << (backwardOpen ? "open" : "closed") << endl;

  // Print forward coordinates with open/closed status.
  cout << "Forward side: ";
  for (size_t i = 0; i < forwardCoords.size(); ++i) cout << "(+" << forwardCoords[i] << ") ";
  cout << (forwardOpen ? "open" : "closed") << endl;

  // Combined group (skipping center):
  cout << "Combined coordinates: ";
  for (size_t i = 0; i < backwardCoords.size(); ++i) cout << "(" << backwardCoords[i] << ") ";
  cout << "(center) ";
  for (size_t i = 0; i < forwardCoords.size(); ++i) cout << "(+" << forwardCoords[i] << ") ";

  // Overall group classification.
  cout << endl;
  if (backwardOpen || forwardOpen)
    cout << "Overall group is open" << endl;
  else
    cout << "Overall group is closed" << endl;

  std::cout << forwardCoords.size() + backwardCoords.size() << std::endl;

  if (forwardOpen && backwardOpen && (forwardCoords.size() + backwardCoords.size() == 2))
    std::cout << "open three" << std::endl;

  if (!(forwardOpen && backwardOpen) && (forwardCoords.size() + backwardCoords.size() == 3))
    std::cout << "closed four" << std::endl;

  if ((forwardOpen && backwardOpen) && (forwardCoords.size() + backwardCoords.size() == 3))
    std::cout << "open four" << std::endl;
  return 0;
}
