#ifndef GOMOKU_HPP
#define GOMOKU_HPP

#include <stdint.h>
// Define constants

#define PLAYER_1 1
#define PLAYER_2 2
#define PLAYER_X 'X'
#define PLAYER_O 'O'
#define OPPONENT(player) ((player) == PLAYER_1 ? PLAYER_2 : PLAYER_1)
#define EMPTY_SPACE 0
#define OUT_OF_BOUNDS 3
#define BOARD_SIZE 19  // Standard Gomoku board size

enum Direction { NORTH = 0, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST };

// Direction vector mapping (x, y) offsets
const int DIRECTIONS[8][2] = {
    {0, -1},  // NORTH
    {1, -1},  // NORTHEAST
    {1, 0},   // EAST
    {1, 1},   // SOUTHEAST
    {0, 1},   // SOUTH
    {-1, 1},  // SOUTHWEST
    {-1, 0},  // WEST
    {-1, -1}  // NORTHWEST
};

/**
 * Helper functions to pack cell values into an unsigned int.
 * Each cell uses 2 bits.
 */
inline unsigned int pack_cells_4(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  return (a << 6) | (b << 4) | (c << 2) | d;
}

inline unsigned int pack_cells_3(unsigned int a, unsigned int b, unsigned int c) {
  return (a << 4) | (b << 2) | c;
}

inline unsigned int pack_cells_2(unsigned int a, unsigned int b) { return (a << 2) | b; }

inline unsigned int pack_cells_1(unsigned int a) { return a; }

/**
 * for zobrist table
 */

static const int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
static const uint64_t LAST_SCORE_MULTIPLIER = 2654435761UL;
static const uint64_t NEXT_SCORE_MULTIPLIER = 1597334677UL;  // Different prime constant

extern uint64_t zobristTable[NUM_CELLS]
                            [3];  // For each cell and state (0: empty, 1: black, 2: white).
extern uint64_t zobristTurn[3];   // For turn: index 1 for BLACK, 2 for WHITE.

void initZobrist();

#endif  // GOMOKU_HPP
