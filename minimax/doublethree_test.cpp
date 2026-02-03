// doublethree_test.cpp
// Consolidated Test Suite for Double Three Logic
// Verifies CForbiddenPointFinder against specific known cases

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <string>

#include "Gomoku.hpp"
#include "Board.hpp"
#include "ForbiddenPointFinder.h"

namespace {

// ============================================================================
// Timing utility
// ============================================================================
double getTimeMs() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// Board printing with coordinates
// ============================================================================
void printBoard(const Board &board, int x_mark = -1, int y_mark = -1) {
  std::cout << "   ";
  for (int x = 0; x < BOARD_SIZE; ++x) {
    std::cout << (char)('A' + x);
  }
  std::cout << "\n";

  for (int y = 0; y < BOARD_SIZE; ++y) {
    if (y + 1 < 10) std::cout << " ";
    std::cout << (y + 1) << " ";
    for (int x = 0; x < BOARD_SIZE; ++x) {
      int cell = board.getValueBit(x, y);
      if (x == x_mark && y == y_mark) {
        std::cout << '*';
      } else if (cell == PLAYER_1) {
        std::cout << 'X';
      } else if (cell == PLAYER_2) {
        std::cout << 'O';
      } else {
        std::cout << '.';
      }
    }
    std::cout << "\n";
  }
}

// ============================================================================
// Helpers
// ============================================================================
void fillFinder(CForbiddenPointFinder &finder, const Board &board, int player) {
  finder.Clear();
  for (int y = 0; y < BOARD_SIZE; ++y) {
    for (int x = 0; x < BOARD_SIZE; ++x) {
      int cell = board.getValueBit(x, y);
      if (cell == EMPTY_SPACE) continue;
      if (cell == player) {
        finder.SetStone(x, y, BLACKSTONE);
      } else {
        finder.SetStone(x, y, WHITESTONE);
      }
    }
  }
}

// ============================================================================
// Specific Tests Section
// ============================================================================
struct TestResult {
  std::string name;
  bool passed;
  double time;
};
std::vector<TestResult> specificResults;

bool runSpecificCase(const std::string& testName, Board &board, int x, int y, int player, bool expectForbidden) {
  CForbiddenPointFinder finder(BOARD_SIZE);
  fillFinder(finder, board, player);

  double t0 = getTimeMs();
  bool result = finder.IsDoubleThree(x, y);
  double t1 = getTimeMs();
  double time = t1 - t0;

  bool passed = (result == expectForbidden);

  if (!passed) {
    std::cout << "\n[FAIL] Test: " << testName << "\n";
    printBoard(board, x, y);
    std::cout << "Target: (" << x << ", " << y << ")\n";
    std::cout << "Expected: " << (expectForbidden ? "FORBIDDEN" : "ALLOWED") << "\n";
    std::cout << "Actual:   " << (result ? "FORBIDDEN" : "ALLOWED") << "\n";
  }

  TestResult tr;
  tr.name = testName;
  tr.passed = passed;
  tr.time = time;
  specificResults.push_back(tr);

  return passed;
}

// --- Test Case Definitions ---

bool test_double_three_basic() {
  Board board;
  // (3,3) should be forbidden
  board.setValueBit(3, 1, PLAYER_1);
  board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(1, 3, PLAYER_1);
  board.setValueBit(2, 3, PLAYER_1);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Basic Double Three", board, 3, 3, PLAYER_1, true);
}

bool test_double_three_edge_patterns() {
  Board board;
  board.setValueBit(4, 3, PLAYER_1);
  board.setValueBit(5, 3, PLAYER_1);
  board.setValueBit(3, 4, PLAYER_1);
  board.setValueBit(3, 5, PLAYER_1);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Edge Patterns (.$OO.)", board, 3, 3, PLAYER_1, true);
}

bool test_double_three_middle1_patterns() {
  Board board;
  board.setValueBit(2, 3, PLAYER_1);
  board.setValueBit(4, 3, PLAYER_1);
  board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(3, 4, PLAYER_1);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Middle1 Patterns (.O$O.)", board, 3, 3, PLAYER_1, true);
}

bool test_double_three_middle2_patterns() {
  Board board;
  board.setValueBit(2, 3, PLAYER_1);
  board.setValueBit(5, 3, PLAYER_1);
  board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(3, 5, PLAYER_1);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Middle2 Patterns (.O$.O.)", board, 3, 3, PLAYER_1, true);
}

bool test_exclusion_edge_1() {
  Board board;
  board.setValueBit(3, 1, PLAYER_1);
  board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(2, 3, PLAYER_2);
  board.setValueBit(4, 3, PLAYER_1);
  board.setValueBit(5, 3, PLAYER_1);
  board.setValueBit(6, 3, PLAYER_2);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Exclusion Edge (X.$OO.X)", board, 3, 3, PLAYER_1, false);
}

bool test_exclusion_edge_2() {
  Board board;
  board.setValueBit(1, 3, PLAYER_1);
  board.setValueBit(4, 3, PLAYER_1);
  board.setValueBit(6, 3, PLAYER_1);
  board.setValueBit(3, 1, PLAYER_1);
  board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Exclusion Edge 2 (O.$O.O)", board, 3, 3, PLAYER_1, false);
}

bool test_exclusion_middle_1() {
  Board board;
  board.setValueBit(3, 1, PLAYER_1);
  board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(0, 3, PLAYER_2);
  board.setValueBit(2, 3, PLAYER_1);
  board.setValueBit(4, 3, PLAYER_1);
  board.setValueBit(6, 3, PLAYER_2);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Exclusion Middle 1 (X.O$O.X)", board, 3, 3, PLAYER_1, false);
}

bool test_exclusion_middle_2() {
  Board board;
  board.setValueBit(3, 1, PLAYER_1);
  board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(2, 3, PLAYER_1);
  board.setValueBit(4, 3, PLAYER_1);
  board.setValueBit(6, 3, PLAYER_1);
  board.setValueBit(8, 8, PLAYER_2);
  return runSpecificCase("Exclusion Middle 2 (O$O.O)", board, 3, 3, PLAYER_1, false);
}

bool test_chimhaha_1() {
  Board board;
  board.setValueBit(6, 5, PLAYER_1);
  board.setValueBit(7, 5, PLAYER_1);
  board.setValueBit(8, 6, PLAYER_1);
  board.setValueBit(7, 7, PLAYER_1);
  board.setValueBit(12, 2, PLAYER_1);
  board.setValueBit(5, 9, PLAYER_2);
  return runSpecificCase("Chimhaha Case 1", board, 9, 5, PLAYER_1, false);
}

bool test_namuwiki_1() {
  Board board;
  // A, B, C, D all forbidden
  board.setValueBit(3, 1, PLAYER_1); board.setValueBit(3, 2, PLAYER_1);
  board.setValueBit(2, 3, PLAYER_1); board.setValueBit(4, 3, PLAYER_1);

  board.setValueBit(1, 9, PLAYER_1); board.setValueBit(4, 9, PLAYER_1);
  board.setValueBit(2, 10, PLAYER_1); board.setValueBit(4, 10, PLAYER_1);

  board.setValueBit(12, 3, PLAYER_1); board.setValueBit(12, 5, PLAYER_1);
  board.setValueBit(9, 6, PLAYER_1); board.setValueBit(13, 6, PLAYER_1);

  board.setValueBit(10, 9, PLAYER_1); board.setValueBit(12, 11, PLAYER_1);
  board.setValueBit(13, 11, PLAYER_1); board.setValueBit(10, 12, PLAYER_1);

  bool passed = true;
  passed &= runSpecificCase("Namuwiki 1 - Point A", board, 3, 3, PLAYER_1, true);
  passed &= runSpecificCase("Namuwiki 1 - Point B", board, 4, 12, PLAYER_1, true);
  passed &= runSpecificCase("Namuwiki 1 - Point C", board, 11, 4, PLAYER_1, true);
  passed &= runSpecificCase("Namuwiki 1 - Point D", board, 10, 11, PLAYER_1, true);
  return passed;
}

bool test_namuwiki_3() {
  Board board;
  board.setValueBit(1, 4, PLAYER_1);
  board.setValueBit(4, 4, PLAYER_1);
  board.setValueBit(5, 4, PLAYER_1);
  board.setValueBit(6, 2, PLAYER_1);
  board.setValueBit(6, 3, PLAYER_1);
  board.setValueBit(8, 4, PLAYER_2);
  return runSpecificCase("Namuwiki 3 (Blocked)", board, 6, 4, PLAYER_1, false);
}

// --------------------------------------------------------

void RunSpecificTests() {
  std::cout << "========================================\n";
  std::cout << "    [1/1] Specific Test Cases\n";
  std::cout << "========================================\n";

  test_double_three_basic();
  test_double_three_edge_patterns();
  test_double_three_middle1_patterns();
  test_double_three_middle2_patterns();
  test_exclusion_edge_1();
  test_exclusion_edge_2();
  test_exclusion_middle_1();
  test_exclusion_middle_2();
  test_chimhaha_1();
  test_namuwiki_1();
  test_namuwiki_3();

  int passed = 0, failed = 0;
  double totalTime = 0;

  for (size_t i = 0; i < specificResults.size(); ++i) {
    const TestResult& r = specificResults[i];
    if (r.passed) ++passed;
    else ++failed;
    totalTime += r.time;
  }

  std::cout << "\nPassed: " << passed << "/" << specificResults.size();
  std::cout << " (Avg: " << (totalTime/specificResults.size()) << "ms)\n\n";

  if (failed > 0) {
    std::cout << "[ERROR] Specific tests failed! Aborting.\n";
    exit(1);
  }
}

}  // namespace

int main() {
  // initZobrist is not strictly needed for ForbiddenPointFinder only,
  // but if Board depends on it for something, good to have.
  // Board constructor might not need it, but let's call it if available.
  initZobrist();

  RunSpecificTests();

  return 0;
}
