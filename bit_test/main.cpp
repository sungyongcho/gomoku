#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>
#include <cassert>

// -----------------------------------------------------------------------------
// Macro Definitions and Constants
// -----------------------------------------------------------------------------

#define BOARD_SIZE 19
#define EMPTY_CELL 0
#define PLAYER_1 1
#define PLAYER_2 2

// For bitwise board representation.
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE) // 361 cells

// -----------------------------------------------------------------------------
// Minimal Board Class
// -----------------------------------------------------------------------------

class Board
{
private:
	// We'll represent the board as a 2D vector of ints.
	std::vector<std::vector<int> > grid;

public:
	Board()
	{
		grid = std::vector<std::vector<int> >(BOARD_SIZE, std::vector<int>(BOARD_SIZE, EMPTY_CELL));
	}

	// Set a stone at (x,y) for a player.
	void setStone(int x, int y, int player)
	{
		if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE)
			grid[y][x] = player;
	}

	// Get the value at (x,y): 0 = empty, 1 = player1, 2 = player2.
	int getValueBit(int x, int y) const
	{
		if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE)
			return -1; // out-of-range (should not be used)
		return grid[y][x];
	}

	// Print the board.
	void print() const
	{
		for (int y = 0; y < BOARD_SIZE; y++)
		{
			for (int x = 0; x < BOARD_SIZE; x++)
			{
				std::cout << grid[y][x] << " ";
			}
			std::cout << "\n";
		}
	}
};

// -----------------------------------------------------------------------------
// Bounding Box Structure
// -----------------------------------------------------------------------------

struct BoundingBox
{
	int minX;
	int minY;
	int maxX;
	int maxY;
};

// -----------------------------------------------------------------------------
// Bitset-Based Move Generation Functions
// -----------------------------------------------------------------------------

// Return a bitset representing occupied cells (union of both players).
std::bitset<TOTAL_CELLS> getOccupiedBitset(const Board &board)
{
	std::bitset<TOTAL_CELLS> occupied;
	for (int y = 0; y < BOARD_SIZE; y++)
	{
		for (int x = 0; x < BOARD_SIZE; x++)
		{
			if (board.getValueBit(x, y) != EMPTY_CELL)
			{
				int idx = y * BOARD_SIZE + x;
				occupied.set(idx);
			}
		}
	}
	return occupied;
}

// Shift functions on the board bitset.
std::bitset<TOTAL_CELLS> shiftLeft(const std::bitset<TOTAL_CELLS> &b)
{
	std::bitset<TOTAL_CELLS> res;
	for (int y = 0; y < BOARD_SIZE; y++)
	{
		for (int x = 1; x < BOARD_SIZE; x++)
		{
			int idx = y * BOARD_SIZE + x;
			if (b.test(idx))
				res.set(y * BOARD_SIZE + (x - 1));
		}
	}
	return res;
}

std::bitset<TOTAL_CELLS> shiftRight(const std::bitset<TOTAL_CELLS> &b)
{
	std::bitset<TOTAL_CELLS> res;
	for (int y = 0; y < BOARD_SIZE; y++)
	{
		for (int x = 0; x < BOARD_SIZE - 1; x++)
		{
			int idx = y * BOARD_SIZE + x;
			if (b.test(idx))
				res.set(y * BOARD_SIZE + (x + 1));
		}
	}
	return res;
}

std::bitset<TOTAL_CELLS> shiftUp(const std::bitset<TOTAL_CELLS> &b)
{
	std::bitset<TOTAL_CELLS> res;
	for (int y = 1; y < BOARD_SIZE; y++)
	{
		for (int x = 0; x < BOARD_SIZE; x++)
		{
			int idx = y * BOARD_SIZE + x;
			if (b.test(idx))
				res.set((y - 1) * BOARD_SIZE + x);
		}
	}
	return res;
}

std::bitset<TOTAL_CELLS> shiftDown(const std::bitset<TOTAL_CELLS> &b)
{
	std::bitset<TOTAL_CELLS> res;
	for (int y = 0; y < BOARD_SIZE - 1; y++)
	{
		for (int x = 0; x < BOARD_SIZE; x++)
		{
			int idx = y * BOARD_SIZE + x;
			if (b.test(idx))
				res.set((y + 1) * BOARD_SIZE + x);
		}
	}
	return res;
}

std::bitset<TOTAL_CELLS> shiftUpLeft(const std::bitset<TOTAL_CELLS> &b)
{
	return shiftLeft(shiftUp(b));
}
std::bitset<TOTAL_CELLS> shiftUpRight(const std::bitset<TOTAL_CELLS> &b)
{
	return shiftRight(shiftUp(b));
}
std::bitset<TOTAL_CELLS> shiftDownLeft(const std::bitset<TOTAL_CELLS> &b)
{
	return shiftLeft(shiftDown(b));
}
std::bitset<TOTAL_CELLS> shiftDownRight(const std::bitset<TOTAL_CELLS> &b)
{
	return shiftRight(shiftDown(b));
}

// Combine all eight directional shifts to get the neighbor mask.
std::bitset<TOTAL_CELLS> neighborMask(const std::bitset<TOTAL_CELLS> &b)
{
	return shiftLeft(b) | shiftRight(b) | shiftUp(b) | shiftDown(b) | shiftUpLeft(b) | shiftUpRight(b) | shiftDownLeft(b) | shiftDownRight(b);
}

// Bit-parallel BFS (flood fill) to get a connected component from a starting cell.
std::bitset<TOTAL_CELLS> floodFill(const std::bitset<TOTAL_CELLS> &occupied, int start)
{
	std::bitset<TOTAL_CELLS> component;
	component.set(start);
	std::bitset<TOTAL_CELLS> prev;
	do
	{
		prev = component;
		component |= neighborMask(component) & occupied;
	} while (component != prev);
	return component;
}

// Get connected components (clusters) as bitsets.
std::vector<std::bitset<TOTAL_CELLS> > getConnectedComponents(const Board &board)
{
	std::vector<std::bitset<TOTAL_CELLS> > components;
	std::bitset<TOTAL_CELLS> occupied = getOccupiedBitset(board);

	while (occupied.any())
	{
		int start = occupied._Find_first(); // GCC extension.
		std::bitset<TOTAL_CELLS> comp = floodFill(occupied, start);
		components.push_back(comp);
		occupied &= ~comp;
	}
	return components;
}

// Compute the bounding box for a connected component.
BoundingBox computeBoundingBox(const std::bitset<TOTAL_CELLS> &comp)
{
	BoundingBox bb;
	bb.minX = BOARD_SIZE;
	bb.minY = BOARD_SIZE;
	bb.maxX = -1;
	bb.maxY = -1;
	for (int idx = 0; idx < TOTAL_CELLS; idx++)
	{
		if (comp.test(idx))
		{
			int x = idx % BOARD_SIZE;
			int y = idx / BOARD_SIZE;
			bb.minX = std::min(bb.minX, x);
			bb.maxX = std::max(bb.maxX, x);
			bb.minY = std::min(bb.minY, y);
			bb.maxY = std::max(bb.maxY, y);
		}
	}
	return bb;
}

// Generate candidate moves using BFS on connected components combined with bounding box and neighborhood filtering.
std::vector<std::pair<int, int> > generateCandidateMovesBFS(const Board &board, int margin = 1)
{
	std::vector<std::pair<int, int> > moves;
	std::vector<std::bitset<TOTAL_CELLS> > components = getConnectedComponents(board);
	std::vector<bool> mark(TOTAL_CELLS, false);

	for (const auto &comp : components)
	{
		BoundingBox bb = computeBoundingBox(comp);
		int minX = std::max(0, bb.minX - margin);
		int minY = std::max(0, bb.minY - margin);
		int maxX = std::min(BOARD_SIZE - 1, bb.maxX + margin);
		int maxY = std::min(BOARD_SIZE - 1, bb.maxY + margin);
		for (int y = minY; y <= maxY; y++)
		{
			for (int x = minX; x <= maxX; x++)
			{
				int idx = y * BOARD_SIZE + x;
				if (!mark[idx] && board.getValueBit(x, y) == EMPTY_CELL)
				{
					bool adjacent = false;
					for (int dx = -1; dx <= 1 && !adjacent; dx++)
					{
						for (int dy = -1; dy <= 1 && !adjacent; dy++)
						{
							int nx = x + dx, ny = y + dy;
							if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE)
							{
								if (board.getValueBit(nx, ny) != EMPTY_CELL)
									adjacent = true;
							}
						}
					}
					if (adjacent)
					{
						moves.push_back(std::make_pair(x, y));
						mark[idx] = true;
					}
				}
			}
		}
	}
	return moves;
}

// -----------------------------------------------------------------------------
// Main Function for Testing
// -----------------------------------------------------------------------------

int main()
{
	// Create a Board instance.
	Board board;

	// Place some stones (simulate a few clusters).
	board.setStone(5, 5, PLAYER_1);
	board.setStone(5, 6, PLAYER_1);
	board.setStone(6, 5, PLAYER_1);

	board.setStone(15, 15, PLAYER_2);
	board.setStone(16, 15, PLAYER_2);
	board.setStone(15, 16, PLAYER_2);

	board.setStone(10, 2, PLAYER_1); // isolated stone

	std::cout << "Board state:\n";
	board.print();

	// Generate candidate moves using BFS connected components approach.
	std::vector<std::pair<int, int> > candidates = generateCandidateMovesBFS(board, 1);

	std::cout << "\nCandidate Moves:\n";
	for (const auto &move : candidates)
	{
		std::cout << "(" << move.first << ", " << move.second << ")\n";
	}

	return 0;
}
