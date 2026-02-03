#include <iostream>
#include <string>

#define SIDE_WINDOW_SIZE 4

// Converts an integer to its 8-bit binary representation as a string.
std::string toBinary8(int side) {
  std::string bin;
  for (int i = 7; i >= 0; --i) {
    bin.push_back((side & (1 << i)) ? '1' : '0');
  }
  return bin;
}

// Converts a binary string (assumed to be of length SIDE_WINDOW_SIZE) to an integer.
int binaryToInt(const std::string& bits) {
  int value = 0;
  for (std::string::size_type i = 0; i < bits.size(); ++i) {
    value = value * 2 + (bits[i] - '0');
  }
  return value;
}

// Slides a window of size SIDE_WINDOW_SIZE over the 8-bit binary representation of 'side'.
// If the window value equals 'player', prints "matches" next to the pattern.
// The parameter 'reverse' determines if the sliding should be in reverse order.
void slideWindow(int side, int player, bool reverse) {
  std::string bin = toBinary8(side);
  int total = bin.size();
  int windowCount = total - SIDE_WINDOW_SIZE + 1;

  if (!reverse) {
    for (int i = 0; i < windowCount; ++i) {
      std::string before = bin.substr(0, i);
      std::string window = bin.substr(i, SIDE_WINDOW_SIZE);
      std::string after = bin.substr(i + SIDE_WINDOW_SIZE);
      std::cout << before << "[" << window << "]" << after;
      if (binaryToInt(window) == player) {
        std::cout << " matches";
      }
      std::cout << std::endl;
    }
  } else {
    for (int i = windowCount - 1; i >= 0; --i) {
      std::string before = bin.substr(0, i);
      std::string window = bin.substr(i, SIDE_WINDOW_SIZE);
      std::string after = bin.substr(i + SIDE_WINDOW_SIZE);
      std::cout << before << "[" << window << "]" << after;
      if (binaryToInt(window) == player) {
        std::cout << " matches";
      }
      std::cout << std::endl;
    }
  }
}

int main() {
  int side = 0xA3;  // Example 8-bit number (163 in decimal)
  int player = 5;   // Example player value (0 to 15 for 4 bits)
  bool reverse = false;

  // Slide in forward direction.
  slideWindow(side, player, reverse);
  std::cout << "\nReverse:" << std::endl;
  // Slide in reverse direction.
  slideWindow(side, player, true);

  return 0;
}
