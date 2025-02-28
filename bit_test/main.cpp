#include <iostream>
#include <stdint.h>

#define BOARD_SIZE 9
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE) // 361 cells
#define UINT64_BITS 64
#define ARRAY_SIZE ((TOTAL_CELLS + UINT64_BITS - 1) / UINT64_BITS) // 6 elements

class Bitboard
{
private:
	uint64_t board[ARRAY_SIZE];

	// Convert (col, row) to bit index
	int getIndex(int col, int row) const
	{
		if (!(0 <= col && col < BOARD_SIZE && 0 <= row && row < BOARD_SIZE))
		{
			std::cout << "Error: index wrong" << std::endl;
			return -1;
		}
		// Fix: Use col and row correctly
		return row * BOARD_SIZE + col;
	}

	// Get the array index and bit position for a given bit index
	void getBitPosition(int index, int &arrayIndex, int &bitPos) const
	{
		arrayIndex = index / UINT64_BITS;
		bitPos = index % UINT64_BITS;
	}

public:
	Bitboard()
	{
		// Initialize board to empty
		for (int i = 0; i < ARRAY_SIZE; ++i)
		{
			board[i] = 0;
		}
	}

	inline int get(int col, int row) const
	{
		// Calculate index directly
		int index = getIndex(col, row);
		if (index < 0)
			return -1;

		// Calculate array index and bit position
		int arrayIndex = index / UINT64_BITS;
		int bitPos = index % UINT64_BITS;

		return (board[arrayIndex] >> bitPos) & 1;
	}

	// Set a bit (place a stone) at (col, row)
	void set(int col, int row)
	{
		int index = getIndex(col, row);
		int arrayIndex, bitPos;
		getBitPosition(index, arrayIndex, bitPos);
		board[arrayIndex] |= ((uint64_t)1 << bitPos);
	}

	// Clear a bit (remove a stone) at (col, row)
	void reset(int col, int row)
	{
		int index = getIndex(col, row);
		int arrayIndex, bitPos;
		getBitPosition(index, arrayIndex, bitPos);
		board[arrayIndex] &= ~((uint64_t)1 << bitPos);
	}

	// Check if a bit is set (is the cell occupied?) at (col, row)
	bool test(int col, int row) const
	{
		int index = getIndex(col, row);
		int arrayIndex, bitPos;
		getBitPosition(index, arrayIndex, bitPos);
		return (board[arrayIndex] & ((uint64_t)1 << bitPos)) != 0;
	}

	// Display the board (for debugging)
	void display(char player) const
	{
		for (int r = 0; r < BOARD_SIZE; ++r)
		{
			for (int c = 0; c < BOARD_SIZE; ++c)
			{
				if (test(c, r)) // Notice the swapped order
				{
					std::cout << " " << player << " ";
				}
				else
				{
					std::cout << " . ";
				}
			}
			std::cout << std::endl;
		}
	}
};

int main()
{
	Bitboard player1_board;
	Bitboard player2_board;

	// Placing stones using (col, row) format
	player1_board.set(4, 0); // Player 1 places at (4, 0)
	player2_board.set(1, 1); // Player 2 places at (1, 1)
	player1_board.set(2, 2); // Player 1 places at (2, 2)
	player2_board.set(3, 3); // Player 2 places at (3, 3)

	std::cout << "Player 1 Board:" << std::endl;
	player1_board.display('X');

	// std::cout << "Player 2 Board:" << std::endl;
	// player2_board.display('O');

	// Check the bit status for (4, 0)
	std::cout << player1_board.get(4, 0) << std::endl;	// Should show that (4, 0) is occupied by Player 1
	// std::cout << player1_board.get(12, 0) << std::endl; // Should show that (4, 0) is occupied by Player 1

	return 0;
}
