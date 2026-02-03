#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>

#define BOARD_SIZE 19
static const int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
static const uint64_t SCORE_MULTIPLIER = 2654435761UL;

// Global Zobrist table for cells: for each cell and each state (0: empty, 1: black, 2: white).
static uint64_t zobristTable[NUM_CELLS][3];
// Global turn keys: for turn information (index 1 for BLACK, index 2 for WHITE).
static uint64_t zobristTurn[3];

// Initialize the Zobrist tables. Call this once at program start.
void initZobrist() {
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < NUM_CELLS; ++i) {
    for (int state = 0; state < 3; ++state) {
      zobristTable[i][state] = ((uint64_t)rand() << 32) | rand();
    }
  }
  zobristTurn[1] = ((uint64_t)rand() << 32) | rand();
  zobristTurn[2] = ((uint64_t)rand() << 32) | rand();
}

// A simple Board class for demonstration.
class Board {
 public:
  int cells[BOARD_SIZE][BOARD_SIZE];
  int nextPlayer;  // 1 for BLACK, 2 for WHITE.
  int last_player_score;
  int next_player_score;

  Board() : nextPlayer(1), last_player_score(0), next_player_score(0) {
    for (int r = 0; r < BOARD_SIZE; r++) {
      for (int c = 0; c < BOARD_SIZE; c++) {
        cells[r][c] = 0;
      }
    }
  }

  int getValueBit(int col, int row) const { return cells[row][col]; }

  void setValueBit(int col, int row, int newState) { cells[row][col] = newState; }

  int getNextPlayer() const { return nextPlayer; }

  uint64_t getHash() const {
    uint64_t hash = 0;
    for (int row = 0; row < BOARD_SIZE; ++row) {
      for (int col = 0; col < BOARD_SIZE; ++col) {
        int state = getValueBit(col, row);
        int index = row * BOARD_SIZE + col;
        hash ^= zobristTable[index][state];
      }
    }
    hash ^= zobristTurn[getNextPlayer()];
    hash ^= static_cast<uint64_t>(last_player_score) * SCORE_MULTIPLIER;
    hash ^= static_cast<uint64_t>(next_player_score) * SCORE_MULTIPLIER;
    return hash;
  }
};

// A simple transposition table entry structure.
struct TTEntry {
  int evaluation;  // Sample evaluation score.
};

typedef std::map<uint64_t, TTEntry> TranspositionTable;
static TranspositionTable transTable;

int main() {
  initZobrist();

  Board board;
  board.setValueBit(3, 3, 1);  // BLACK at (3,3)
  board.setValueBit(4, 3, 2);  // WHITE at (4,3)
  board.setValueBit(3, 4, 1);  // BLACK at (3,4)

  board.nextPlayer = 2;  // Next move by WHITE.
  board.last_player_score = 2;
  board.next_player_score = 3;

  uint64_t hash = board.getHash();
  std::cout << "Board hash: " << hash << std::endl;

  if (transTable.find(hash) != transTable.end()) {
    std::cout << "Transposition table entry found. Evaluation: " << transTable[hash].evaluation
              << std::endl;
  } else {
    std::cout << "No entry found for this board state. Storing sample evaluation (42)."
              << std::endl;
    transTable[hash] = TTEntry{42};
  }

  return 0;
}
