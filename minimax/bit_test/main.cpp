#include <stdint.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <ctime>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

//////////////////////////////
// Macro Definitions
//////////////////////////////

#define BOARD_SIZE 19
#define EMPTY_CELL 0
#define PLAYER_1 1
#define PLAYER_2 2

// Total cells on board (19 * 19 = 361)
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE)

// For row-based board, each row is stored in a uint64_t and only the lower BOARD_SIZE bits are
// used.
static const uint64_t rowMask = ((uint64_t)1 << BOARD_SIZE) - 1;

//////////////////////////////
// Board Class (Row-Based Representation)
//////////////////////////////

class Board {
 public:
  // Each row is stored in a uint64_t; only the lower BOARD_SIZE bits are used.
  uint64_t last_player_board[BOARD_SIZE];
  uint64_t next_player_board[BOARD_SIZE];

  Board() {
    for (int i = 0; i < BOARD_SIZE; i++) {
      last_player_board[i] = 0;
      next_player_board[i] = 0;
    }
  }

  // Check if (col, row) is within board boundaries.
  bool isValidCoordinate(int col, int row) const {
    return (col >= 0 && col < BOARD_SIZE && row >= 0 && row < BOARD_SIZE);
  }

  // Set a stone at (col, row) for a given player.
  void setValueBit(int col, int row, int player) {
    if (!isValidCoordinate(col, row)) return;
    uint64_t mask = 1ULL << col;
    // Clear cell in both boards.
    last_player_board[row] &= ~mask;
    next_player_board[row] &= ~mask;
    if (player == PLAYER_1)
      last_player_board[row] |= mask;
    else if (player == PLAYER_2)
      next_player_board[row] |= mask;
  }

  // Get the value at (col, row): returns PLAYER_1, PLAYER_2, or EMPTY_CELL.
  int getValueBit(int col, int row) const {
    if (!isValidCoordinate(col, row)) return EMPTY_CELL;
    uint64_t mask = 1ULL << col;
    if (last_player_board[row] & mask) return PLAYER_1;
    if (next_player_board[row] & mask) return PLAYER_2;
    return EMPTY_CELL;
  }

  // Print the board.
  void print() const {
    for (int r = 0; r < BOARD_SIZE; r++) {
      for (int c = 0; c < BOARD_SIZE; c++) {
        int val = getValueBit(c, r);
        if (val == PLAYER_1)
          cout << "1 ";
        else if (val == PLAYER_2)
          cout << "2 ";
        else
          cout << ". ";
      }
      cout << "\n";
    }
  }
};

//////////////////////////////
// Candidate Move Generation: Row-Based Neighbor Mask
//////////////////////////////

// Compute occupancy for each row: union of both players.
void getOccupancy(const Board &board, uint64_t occupancy[BOARD_SIZE]) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    occupancy[i] = board.last_player_board[i] | board.next_player_board[i];
  }
}

// Horizontal shifts for a row.
inline uint64_t shiftRowLeft(uint64_t row) { return (row << 1) & rowMask; }
inline uint64_t shiftRowRight(uint64_t row) { return row >> 1; }

// Compute neighbor mask for each row based on occupancy.
void computeNeighborMask(const uint64_t occupancy[BOARD_SIZE], uint64_t neighbor[BOARD_SIZE]) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    uint64_t row = occupancy[i];
    uint64_t horz = shiftRowLeft(row) | shiftRowRight(row) | row;
    uint64_t vert = 0;
    if (i > 0) vert |= occupancy[i - 1];
    if (i < BOARD_SIZE - 1) vert |= occupancy[i + 1];
    uint64_t diag = 0;
    if (i > 0) {
      diag |= shiftRowLeft(occupancy[i - 1]);
      diag |= shiftRowRight(occupancy[i - 1]);
    }
    if (i < BOARD_SIZE - 1) {
      diag |= shiftRowLeft(occupancy[i + 1]);
      diag |= shiftRowRight(occupancy[i + 1]);
    }
    neighbor[i] = horz | vert | diag;
  }
}

// Generate candidate moves using row-based neighbor mask.
vector<pair<int, int> > generateCandidateMoves(const Board &board) {
  vector<pair<int, int> > moves;
  uint64_t occupancy[BOARD_SIZE];
  uint64_t neighbor[BOARD_SIZE];
  getOccupancy(board, occupancy);
  computeNeighborMask(occupancy, neighbor);

  for (int row = 0; row < BOARD_SIZE; row++) {
    uint64_t candidates = neighbor[row] & (~occupancy[row]) & rowMask;
    for (int col = 0; col < BOARD_SIZE; col++) {
      if (candidates & (1ULL << col)) moves.push_back(make_pair(col, row));
    }
  }
  return moves;
}

//////////////////////////////
// Candidate Move Generation: Bitset-Based BFS (Contiguous Representation)
//////////////////////////////

// Convert board occupancy to a contiguous bitset (361 bits).
std::bitset<TOTAL_CELLS> getOccupiedBitset(const Board &board) {
  std::bitset<TOTAL_CELLS> occ;
  for (int r = 0; r < BOARD_SIZE; r++) {
    for (int c = 0; c < BOARD_SIZE; c++) {
      if (board.getValueBit(c, r) != EMPTY_CELL) {
        int idx = r * BOARD_SIZE + c;
        occ.set(idx);
      }
    }
  }
  return occ;
}

// Shift functions for a contiguous bitset.
std::bitset<TOTAL_CELLS> shiftLeft(const std::bitset<TOTAL_CELLS> &b) {
  std::bitset<TOTAL_CELLS> res;
  for (int r = 0; r < BOARD_SIZE; r++) {
    for (int c = 1; c < BOARD_SIZE; c++) {
      int idx = r * BOARD_SIZE + c;
      if (b.test(idx)) res.set(r * BOARD_SIZE + (c - 1));
    }
  }
  return res;
}

std::bitset<TOTAL_CELLS> shiftRight(const std::bitset<TOTAL_CELLS> &b) {
  std::bitset<TOTAL_CELLS> res;
  for (int r = 0; r < BOARD_SIZE; r++) {
    for (int c = 0; c < BOARD_SIZE - 1; c++) {
      int idx = r * BOARD_SIZE + c;
      if (b.test(idx)) res.set(r * BOARD_SIZE + (c + 1));
    }
  }
  return res;
}

std::bitset<TOTAL_CELLS> shiftUp(const std::bitset<TOTAL_CELLS> &b) {
  std::bitset<TOTAL_CELLS> res;
  for (int r = 1; r < BOARD_SIZE; r++) {
    for (int c = 0; c < BOARD_SIZE; c++) {
      int idx = r * BOARD_SIZE + c;
      if (b.test(idx)) res.set((r - 1) * BOARD_SIZE + c);
    }
  }
  return res;
}

std::bitset<TOTAL_CELLS> shiftDown(const std::bitset<TOTAL_CELLS> &b) {
  std::bitset<TOTAL_CELLS> res;
  for (int r = 0; r < BOARD_SIZE - 1; r++) {
    for (int c = 0; c < BOARD_SIZE; c++) {
      int idx = r * BOARD_SIZE + c;
      if (b.test(idx)) res.set((r + 1) * BOARD_SIZE + c);
    }
  }
  return res;
}

std::bitset<TOTAL_CELLS> neighborMask(const std::bitset<TOTAL_CELLS> &b) {
  return shiftLeft(b) | shiftRight(b) | shiftUp(b) | shiftDown(b) | shiftLeft(shiftUp(b)) |
         shiftRight(shiftUp(b)) | shiftLeft(shiftDown(b)) | shiftRight(shiftDown(b));
}

// Flood fill (bit-parallel BFS) to extract a connected component from a starting cell.
std::bitset<TOTAL_CELLS> floodFill(const std::bitset<TOTAL_CELLS> &occupied, int start) {
  std::bitset<TOTAL_CELLS> comp;
  comp.set(start);
  std::bitset<TOTAL_CELLS> prev;
  do {
    prev = comp;
    comp |= neighborMask(comp) & occupied;
  } while (comp != prev);
  return comp;
}

// Get connected components (clusters) as a vector of bitsets.
vector<std::bitset<TOTAL_CELLS> > getConnectedComponents(const Board &board) {
  vector<std::bitset<TOTAL_CELLS> > comps;
  std::bitset<TOTAL_CELLS> occ = getOccupiedBitset(board);
  while (occ.any()) {
    int start = occ._Find_first();
    std::bitset<TOTAL_CELLS> comp = floodFill(occ, start);
    comps.push_back(comp);
    occ &= ~comp;
  }
  return comps;
}

// Bounding box structure.
struct BoundingBox {
  int minX, minY, maxX, maxY;
};

// Compute bounding box for a connected component.
BoundingBox computeBoundingBox(const std::bitset<TOTAL_CELLS> &comp) {
  BoundingBox bb;
  bb.minX = BOARD_SIZE;
  bb.minY = BOARD_SIZE;
  bb.maxX = -1;
  bb.maxY = -1;
  for (int idx = 0; idx < TOTAL_CELLS; idx++) {
    if (comp.test(idx)) {
      int x = idx % BOARD_SIZE;
      int y = idx / BOARD_SIZE;
      bb.minX = min(bb.minX, x);
      bb.maxX = max(bb.maxX, x);
      bb.minY = min(bb.minY, y);
      bb.maxY = max(bb.maxY, y);
    }
  }
  return bb;
}

// Generate candidate moves using the bitset-based BFS approach.
// For each connected component, compute its bounding box (expanded by margin)
// and then collect empty cells within that box that have at least one adjacent stone.
vector<pair<int, int> > generateCandidateMovesBFS(const Board &board, int margin = 1) {
  vector<pair<int, int> > moves;
  vector<bitset<TOTAL_CELLS> > comps = getConnectedComponents(board);
  vector<bool> mark(TOTAL_CELLS, false);

  for (size_t k = 0; k < comps.size(); k++) {
    BoundingBox bb = computeBoundingBox(comps[k]);
    int minX = max(0, bb.minX - margin);
    int minY = max(0, bb.minY - margin);
    int maxX = min(BOARD_SIZE - 1, bb.maxX + margin);
    int maxY = min(BOARD_SIZE - 1, bb.maxY + margin);
    for (int y = minY; y <= maxY; y++) {
      for (int x = minX; x <= maxX; x++) {
        int idx = y * BOARD_SIZE + x;
        if (!mark[idx] && board.getValueBit(x, y) == EMPTY_CELL) {
          bool adjacent = false;
          for (int dx = -1; dx <= 1 && !adjacent; dx++) {
            for (int dy = -1; dy <= 1 && !adjacent; dy++) {
              int nx = x + dx, ny = y + dy;
              if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                if (board.getValueBit(nx, ny) != EMPTY_CELL) adjacent = true;
              }
            }
          }
          if (adjacent) {
            moves.push_back(make_pair(x, y));
            mark[idx] = true;
          }
        }
      }
    }
  }
  return moves;
}

//////////////////////////////
// Timing Helper Functions
//////////////////////////////

clock_t getCurrentTime() { return clock(); }

double ticksToMilliseconds(clock_t ticks) { return (double)ticks * 1000.0 / CLOCKS_PER_SEC; }

double ticksToNanoseconds(clock_t ticks) { return (double)ticks * 1e9 / CLOCKS_PER_SEC; }

//////////////////////////////
// Print Board with Candidate Moves Marked as 'C'
//////////////////////////////

void printBoardWithCandidates(const Board &board, const vector<pair<int, int> > &candidates) {
  // Create a 2D display grid.
  vector<vector<char> > display(BOARD_SIZE, vector<char>(BOARD_SIZE, '.'));
  for (int y = 0; y < BOARD_SIZE; y++) {
    for (int x = 0; x < BOARD_SIZE; x++) {
      int val = board.getValueBit(x, y);
      if (val == PLAYER_1)
        display[y][x] = '1';
      else if (val == PLAYER_2)
        display[y][x] = '2';
    }
  }
  // Mark candidate moves with 'C' (if the cell is empty).
  for (size_t i = 0; i < candidates.size(); i++) {
    int x = candidates[i].first, y = candidates[i].second;
    if (board.getValueBit(x, y) == EMPTY_CELL) display[y][x] = 'C';
  }
  // Print the board.
  for (int y = 0; y < BOARD_SIZE; y++) {
    for (int x = 0; x < BOARD_SIZE; x++) {
      cout << display[y][x] << " ";
    }
    cout << "\n";
  }
}

//////////////////////////////
// Main Function for Testing and Timing
//////////////////////////////

int main() {
  Board board;

  // Place stones to simulate clusters.
  // Cluster 1: PLAYER_1 cluster.
  board.setValueBit(5, 5, PLAYER_1);
  board.setValueBit(5, 6, PLAYER_1);
  board.setValueBit(6, 5, PLAYER_1);

  // Cluster 2: PLAYER_2 cluster.
  board.setValueBit(15, 15, PLAYER_2);
  board.setValueBit(16, 15, PLAYER_2);
  board.setValueBit(15, 16, PLAYER_2);

  // Isolated stone: PLAYER_1.
  board.setValueBit(10, 2, PLAYER_1);

  board.setValueBit(0, 0, PLAYER_2);

  // middle space.
  board.setValueBit(15, 4, PLAYER_1);
  board.setValueBit(14, 5, PLAYER_1);
  board.setValueBit(15, 6, PLAYER_1);
  board.setValueBit(16, 5, PLAYER_1);

  cout << "Board state:" << endl;
  board.print();

  // Time candidate move generation using row-based neighbor mask.
  clock_t start1 = getCurrentTime();
  vector<pair<int, int> > candidates1 = generateCandidateMoves(board);
  clock_t end1 = getCurrentTime();
  double ms1 = ticksToMilliseconds(end1 - start1);
  double ns1 = ticksToNanoseconds(end1 - start1);

  cout << "\nCandidate Moves (Row-based):" << endl;
  for (size_t i = 0; i < candidates1.size(); i++) {
    cout << "(" << candidates1[i].first << ", " << candidates1[i].second << ")" << endl;
  }
  cout << "Row-based generation time: " << ms1 << " ms (" << ns1 << " ns)" << endl;

  // Time candidate move generation using bitset-based BFS approach.
  clock_t start2 = getCurrentTime();
  vector<pair<int, int> > candidates2 = generateCandidateMovesBFS(board, 1);
  clock_t end2 = getCurrentTime();
  double ms2 = ticksToMilliseconds(end2 - start2);
  double ns2 = ticksToNanoseconds(end2 - start2);

  cout << "\nCandidate Moves (Bitset-based BFS):" << endl;
  for (size_t i = 0; i < candidates2.size(); i++) {
    cout << "(" << candidates2[i].first << ", " << candidates2[i].second << ")" << endl;
  }
  cout << "Bitset-based BFS generation time: " << ms2 << " ms (" << ns2 << " ns)" << endl;

  // Print board with candidate moves marked as 'C' for row-based method.
  cout << "\nBoard with Candidate Moves (Row-based marked as 'C'):" << endl;
  printBoardWithCandidates(board, candidates1);

  // Print board with candidate moves marked as 'C' for bitset-based BFS method.
  cout << "\nBoard with Candidate Moves (Bitset-based BFS marked as 'C'):" << endl;
  printBoardWithCandidates(board, candidates2);

  return 0;
}
