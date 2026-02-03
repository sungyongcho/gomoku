#include <iostream>
using namespace std;

#define SIDE_WINDOW_SIZE 4
#define COMBINED_WINDOW_SIZE (2 * SIDE_WINDOW_SIZE + 1)
#define CENTER_VALUE 0

// Stub evaluation function: for testing, just return the pattern.
int evaluateCombinedPattern(unsigned int pattern, int player) { return pattern; }
void printPattern(unsigned int pattern, int numCells) {
  for (int i = 0; i < numCells; ++i) {
    // Calculate shift so that the leftmost cell is printed first.
    int shift = 2 * (numCells - 1 - i);
    int cell = (pattern >> shift) & 0x3;
    cout << cell << " ";
  }
  cout << endl;
}
// Left side validation:
// The left side (SIDE_WINDOW_SIZE cells) is represented so that the outermost cell
// is in the highest order bits. Validity rule:
// - OUT_OF_BOUNDS (3) cells must appear only contiguously from the outer edge.
// - Once a cell is not 3, no later cell may be 3.
bool isValidLeftSidePattern(unsigned int sidePattern) {
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
bool isValidRightSidePattern(unsigned int sidePattern) {
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

int main() {
  // Total number of possible side patterns (each cell uses 2 bits).
  const unsigned int sideCount = 1 << (2 * SIDE_WINDOW_SIZE);
  unsigned int validCount = 0;

  // Iterate over all possible left and right side patterns.
  for (unsigned int left = 0; left < sideCount; ++left) {
    if (!isValidLeftSidePattern(left)) continue;
    for (unsigned int right = 0; right < sideCount; ++right) {
      if (!isValidRightSidePattern(right)) continue;

      // Build the full pattern:
      // Left side occupies the highest 2*SIDE_WINDOW_SIZE bits,
      // then the fixed center (2 bits),
      // and then the right side occupies the lowest 2*SIDE_WINDOW_SIZE bits.
      unsigned int pattern =
          (left << (2 * (SIDE_WINDOW_SIZE + 1))) | (CENTER_VALUE << (2 * SIDE_WINDOW_SIZE)) | right;

      // Evaluate the pattern for PLAYER_1 and PLAYER_2 (stubbed here).
      int score1 = evaluateCombinedPattern(pattern, 1);
      int score2 = evaluateCombinedPattern(pattern, 2);

      validCount++;
      // For demonstration, print the first few valid patterns.
      printPattern(pattern, 9);
    }
  }

  cout << "Total valid patterns: " << validCount << endl;

  return 0;
}
