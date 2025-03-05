#include <iostream>
#include <stdint.h>
#include <string.h>
#include <ctime>
#include <chrono>


// --- Constants ---
#define BOARD_SIZE 19
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE)
#define UINT64_BITS 64
#define ARRAY_SIZE ((TOTAL_CELLS + UINT64_BITS - 1) / UINT64_BITS)
#define PLAYER_1 1
#define PLAYER_2 2
#define EMPTY_SPACE 0

// --- Bitboard Class ---
// We maintain two bitboards: board1 for PLAYER_1 and board2 for PLAYER_2.
class Bitboard
{
public:
	uint64_t board1[ARRAY_SIZE];
	uint64_t board2[ARRAY_SIZE];

	Bitboard()
	{
		for (int i = 0; i < ARRAY_SIZE; i++)
		{
			board1[i] = 0;
			board2[i] = 0;
		}
	}

	// Copy constructor for benchmarking.
	Bitboard(const Bitboard &other)
	{
		memcpy(board1, other.board1, sizeof(board1));
		memcpy(board2, other.board2, sizeof(board2));
	}

	// Convert (col, row) to a 1D index; returns -1 if out-of-bounds.
	inline int getIndex(int col, int row) const
	{
		if (col < 0 || col >= BOARD_SIZE || row < 0 || row >= BOARD_SIZE)
			return -1;
		return row * BOARD_SIZE + col;
	}

	// Set the cell (col, row) to a given player.
	inline void set_value(int col, int row, int player)
	{
		int idx = getIndex(col, row);
		if (idx < 0)
			return;
		int word = idx / UINT64_BITS;
		int bit = idx % UINT64_BITS;
		uint64_t mask = (uint64_t)1 << bit;
		// Clear cell in both boards.
		board1[word] &= ~mask;
		board2[word] &= ~mask;
		if (player == PLAYER_1)
			board1[word] |= mask;
		else if (player == PLAYER_2)
			board2[word] |= mask;
	}

	// Get the value at (col, row).
	inline int get_value(int col, int row) const
	{
		int idx = getIndex(col, row);
		if (idx < 0)
			return EMPTY_SPACE;
		int word = idx / UINT64_BITS;
		int bit = idx % UINT64_BITS;
		uint64_t mask = (uint64_t)1 << bit;
		if (board1[word] & mask)
			return PLAYER_1;
		if (board2[word] & mask)
			return PLAYER_2;
		return EMPTY_SPACE;
	}

	// Print board (for debugging). X = PLAYER_1, O = PLAYER_2, . = EMPTY
	void print_board() const
	{
		for (int r = 0; r < BOARD_SIZE; r++)
		{
			for (int c = 0; c < BOARD_SIZE; c++)
			{
				int v = get_value(c, r);
				if (v == PLAYER_1)
					std::cout << "X ";
				else if (v == PLAYER_2)
					std::cout << "O ";
				else
					std::cout << ". ";
			}
			std::cout << std::endl;
		}
	}
};

// --- DFS-Based Capture Check ---
// Recursively checks for a capture pattern starting at (x,y)
// Direction is given by (dx, dy). Count starts at 1 for the first adjacent cell.
bool dfs_capture(Bitboard &board, int x, int y, int currentPlayer, int dx, int dy, int count)
{
	int nx = x + dx;
	int ny = y + dy;
	if (count == 3)
	{
		// For the third cell, we expect currentPlayer's stone.
		return board.get_value(nx, ny) == currentPlayer;
	}
	int opponent = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;
	if (board.get_value(nx, ny) != opponent)
		return false;
	return dfs_capture(board, nx, ny, currentPlayer, dx, dy, count + 1);
}

// DFS version: check and apply capture in one direction for last move at (x, y).
// Pattern: starting from the last move, check the three cells in direction (dx, dy).
bool dfs_check_and_apply_capture(Bitboard &board, int x, int y, int currentPlayer, int dx, int dy)
{
	if (dfs_capture(board, x, y, currentPlayer, dx, dy, 1))
	{
		// Remove captured opponent stones (cells at (x+dx,y+dy) and (x+2*dx,y+2*dy)).
		board.set_value(x + dx, y + dy, EMPTY_SPACE);
		board.set_value(x + 2 * dx, y + 2 * dy, EMPTY_SPACE);
		return true;
	}
	return false;
}

// --- Bitmask-Based Capture Check ---
// Uses direct bit-level operations to check the local pattern.
bool bitmask_check_and_apply_capture(Bitboard &board, int x, int y, int currentPlayer, int dx, int dy)
{
	int opponent = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;

	// Use the bitboard arrays directly.
	uint64_t *P = (currentPlayer == PLAYER_1) ? board.board1 : board.board2;
	uint64_t *O = (currentPlayer == PLAYER_1) ? board.board2 : board.board1;

	// Compute indices for the three cells in the capture pattern.
	int idx1 = board.getIndex(x + dx, y + dy);
	int idx2 = board.getIndex(x + 2 * dx, y + 2 * dy);
	int idx3 = board.getIndex(x + 3 * dx, y + 3 * dy);

	// If any index is out-of-bounds, pattern cannot occur.
	if (idx1 < 0 || idx2 < 0 || idx3 < 0)
		return false;

	int w1 = idx1 / UINT64_BITS, b1 = idx1 % UINT64_BITS;
	int w2 = idx2 / UINT64_BITS, b2 = idx2 % UINT64_BITS;
	int w3 = idx3 / UINT64_BITS, b3 = idx3 % UINT64_BITS;

	uint64_t mask1 = (uint64_t)1 << b1;
	uint64_t mask2 = (uint64_t)1 << b2;
	uint64_t mask3 = (uint64_t)1 << b3;

	// Check the pattern:
	// Cells at idx1 and idx2 must have opponent stones, and cell at idx3 must have current player's stone.
	bool pattern = (O[w1] & mask1) && (O[w2] & mask2) && (P[w3] & mask3);

	if (pattern)
	{
		// Remove the captured opponent stones.
		O[w1] &= ~mask1;
		O[w2] &= ~mask2;
		return true;
	}
	return false;
}

// --- Main ---
// Set up a capture pattern and run each method many times to measure processing time.
int main()
{
	int directions[8][2] = {
		{0, -1}, // NORTH
		{1, -1}, // NORTHEAST
		{1, 0},	 // EAST
		{1, 1},	 // SOUTHEAST
		{0, 1},	 // SOUTH
		{-1, 1}, // SOUTHWEST
		{-1, 0}, // WEST
		{-1, -1} // NORTHWEST
	};
	// Create a board and set up a capture pattern horizontally:
	// We create a pattern: X, O, O, X along row 5 starting at column 3.
	Bitboard board;
	int currentPlayer = PLAYER_1;
	int opponent = PLAYER_2;
	int row = 5, col = 3;
	// Choose a last move that is central.
	int x = 10, y = 10;

	// Create a board and set the last move.
	board.set_value(x, y, currentPlayer);

	// For each direction, set up the capture pattern:
	// (x+dx, y+dy) and (x+2*dx, y+2*dy) are opponent stones,
	// (x+3*dx, y+3*dy) is currentPlayer's stone.
	for (int d = 0; d < 8; d++)
	{
		int dx = directions[d][0], dy = directions[d][1];
		board.set_value(x + dx, y + dy, opponent);
		board.set_value(x + 2 * dx, y + 2 * dy, opponent);
		board.set_value(x + 3 * dx, y + 3 * dy, currentPlayer);
	}

	auto start = std::chrono::high_resolution_clock::now();
	// ... code to benchmark ...
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

	// Benchmark DFS method.
	Bitboard dfsBoard(board); // Copy the original board for each iteration.
	clock_t start_dfs = clock();
	for (int d = 0; d < 8; d++)
	{
		int dx = directions[d][0], dy = directions[d][1];
		dfs_check_and_apply_capture(dfsBoard, x, y, currentPlayer, dx, dy);
	}
	clock_t end_dfs = clock();
	double time_dfs = double(end_dfs - start_dfs) / CLOCKS_PER_SEC * 1000.0;

	// Benchmark Bitmask method.
	Bitboard bitmaskBoard(board); // Copy the original board for each iteration.
	clock_t start_bitmask = clock();
	for (int d = 0; d < 8; d++)
	{
		int dx = directions[d][0], dy = directions[d][1];
		bitmask_check_and_apply_capture(bitmaskBoard, x, y, currentPlayer, dx, dy);
	}
	clock_t end_bitmask = clock();
	double time_bitmask = double(end_bitmask - start_bitmask) / CLOCKS_PER_SEC * 1000.0;

	// Print processing times.
	std::cout << "DFS capture method time: " << time_dfs << " ms" << std::endl;
	std::cout << "Bitmask capture method time: " << time_bitmask << " ms" << std::endl;

	// Also show board state after one capture (for visual confirmation)
	std::cout << "\nInitial board:" << std::endl;
	board.print_board();

	std::cout << "\nAfter dfs board:" << std::endl;
	dfsBoard.print_board();

	std::cout << "\nAfter bitmask board:" << std::endl;
	bitmaskBoard.print_board();

	return 0;
}
