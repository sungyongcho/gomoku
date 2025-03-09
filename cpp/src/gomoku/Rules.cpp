
#include "Rules.hpp"

enum Direction
{
	NORTH = 0,
	NORTHEAST,
	EAST,
	SOUTHEAST,
	SOUTH,
	SOUTHWEST,
	WEST,
	NORTHWEST
};

// Direction vector mapping (x, y) offsets
const int DIRECTIONS[8][2] = {
	{0, -1}, // NORTH
	{1, -1}, // NORTHEAST
	{1, 0},	 // EAST
	{1, 1},	 // SOUTHEAST
	{0, 1},	 // SOUTH
	{-1, 1}, // SOUTHWEST
	{-1, 0}, // WEST
	{-1, -1} // NORTHWEST
};

// Bitmask-based capture check that stores captured stone coordinates.
// It checks for the pattern: opponent stone at (x+dx, y+dy) and (x+2*dx, y+2*dy),
// with currentPlayer’s stone at (x+3*dx, y+3*dy). If the pattern is found,
// it removes the captured stones from the opponent's bitboard and adds their coordinates
// to the captured vector.
bool bitmask_check_and_apply_capture(Board &board, int x, int y, int currentPlayer, int dx, int dy,
									 std::vector<std::pair<int, int> > &captured)
{
	int opponent = (currentPlayer == PLAYER_1) ? PLAYER_2 : PLAYER_1;

	// Use the bitboard arrays directly.
	uint64_t *P = board.get_bitboard_by_player(currentPlayer); // e.g., returns board.board1 if PLAYER_1
	uint64_t *O = board.get_bitboard_by_player(opponent);

	// Compute indices for the three cells in the capture pattern.
	int idx1 = board.getIndex(x + dx, y + dy);
	int idx2 = board.getIndex(x + 2 * dx, y + 2 * dy);
	int idx3 = board.getIndex(x + 3 * dx, y + 3 * dy);

	// If any index is out-of-bounds, the pattern cannot occur.
	if (idx1 < 0 || idx2 < 0 || idx3 < 0)
		return false;

	int w1 = idx1 / UINT64_BITS, b1 = idx1 % UINT64_BITS;
	int w2 = idx2 / UINT64_BITS, b2 = idx2 % UINT64_BITS;
	int w3 = idx3 / UINT64_BITS, b3 = idx3 % UINT64_BITS;

	uint64_t mask1 = (uint64_t)1 << b1;
	uint64_t mask2 = (uint64_t)1 << b2;
	uint64_t mask3 = (uint64_t)1 << b3;

	// Check the pattern:
	// Cells at idx1 and idx2 must contain opponent stones,
	// and the cell at idx3 must contain the current player's stone.
	bool pattern = ((O[w1] & mask1) != 0) && ((O[w2] & mask2) != 0) && ((P[w3] & mask3) != 0);

	if (pattern)
	{
		// Remove the captured opponent stones.
		O[w1] &= ~mask1;
		O[w2] &= ~mask2;
		// Store captured stone coordinates.
		captured.push_back(std::make_pair(x + dx, y + dy));
		captured.push_back(std::make_pair(x + 2 * dx, y + 2 * dy));
		return true;
	}
	return false;
}

bool Rules::get_captured_stones_bit(Board &board, int x, int y, const std::string &last_player,
									std::vector<std::pair<int, int> > &captured)
{
	int currentPlayer = (last_player == "X") ? PLAYER_1 : PLAYER_2;
	bool check = false;
	// Loop over the 8 directions.
	for (size_t i = 0; i < 8; ++i)
	{
		// Use the direction offsets directly.
		// If a capture is found in this direction, mark check true.
		if (bitmask_check_and_apply_capture(board, x, y, currentPlayer, DIRECTIONS[i][0], DIRECTIONS[i][1], captured))
			check = true;
	}
	return check;
}

namespace
{

	// Check if (x+offset_x, y+offset_y) is within board bounds.
	bool is_within_bounds(int x, int y, int offset_x, int offset_y)
	{
		int new_x = x + offset_x;
		int new_y = y + offset_y;
		return (new_x >= 0 && new_x < BOARD_SIZE && new_y >= 0 && new_y < BOARD_SIZE);
	}

	// Extract a “line” (vector of cell values) of given length in direction (dx, dy)
	// starting from one step away from (x, y). Returns an empty vector if out‐of‐bounds.
	std::vector<int> get_line(Board &board, int x, int y, int dx, int dy, int length)
	{
		std::vector<int> result;
		for (int i = 1; i <= length; ++i)
		{
			if (!is_within_bounds(x, y, dx * i, dy * i))
				return std::vector<int>(); // empty
			int new_x = x + dx * i;
			int new_y = y + dy * i;
			result.push_back(board.get_value(new_x, new_y));
		}
		return result;
	}

	// Check for an “open three” with a middle gap pattern.
	bool check_middle(Board &board, int x, int y, int dx, int dy, int player, int opponent)
	{
		std::vector<int> line = get_line(board, x, y, dx, dy, 3);
		std::vector<int> line_opp = get_line(board, x, y, -dx, -dy, 3);
		if (line.size() < 3 || line_opp.size() < 3)
			return false;

		// Case 1:
		bool cond1 = (line_opp[0] == player && line_opp[1] == EMPTY_SPACE && line_opp[2] == opponent &&
					  line[0] == player && line[1] == EMPTY_SPACE && line[2] == opponent);
		bool cond2 = (line_opp[0] == player &&
					  line[0] == player && line[1] == EMPTY_SPACE && line[2] == player);
		bool cond3 = (line_opp[0] == player && line_opp[1] == EMPTY_SPACE &&
					  line[0] == player && line[1] == EMPTY_SPACE);
		if (!(cond1 || cond2) && cond3)
			return true;

		// Case 2:
		bool cond4 = (line_opp[0] == player && line_opp[1] == EMPTY_SPACE);
		bool cond5 = (line[0] == EMPTY_SPACE && line[1] == player && line[2] == EMPTY_SPACE);
		if (cond4 && cond5)
			return true;

		return false;
	}

	// Check for an “open three” pattern on the edge.
	bool check_edge(Board &board, int x, int y, int dx, int dy, int player, int opponent)
	{
		std::vector<int> line = get_line(board, x, y, dx, dy, 4);
		std::vector<int> line_opp = get_line(board, x, y, -dx, -dy, 2);
		if (line.size() < 4 || line_opp.size() < 2)
			return false;

		// Case 1: (e.g., pattern: [player, player, EMPTY_SPACE, opponent] with an opposing EMPTY_SPACE)
		bool cond1a = (line[0] == player && line[1] == player && line[2] == EMPTY_SPACE && line[3] == opponent &&
					   line_opp[0] == EMPTY_SPACE && line_opp[1] == opponent);
		bool cond1b = (line[0] == player && line[1] == player && line[2] == EMPTY_SPACE && line[3] == player);
		if (!(cond1a || cond1b) &&
			(line[0] == player && line[1] == player && line[2] == EMPTY_SPACE && line_opp[0] == EMPTY_SPACE))
			return true;

		// Case 2: (e.g., pattern: [player, EMPTY_SPACE, player, EMPTY_SPACE])
		bool cond2a = (line[0] == player && line[1] == EMPTY_SPACE && line[2] == player &&
					   line_opp[0] == EMPTY_SPACE && line_opp[1] == player);
		if (!cond2a &&
			(line[0] == player && line[1] == EMPTY_SPACE && line[2] == player && line[3] == EMPTY_SPACE && line_opp[0] == EMPTY_SPACE))
			return true;

		// Case 3: (e.g., pattern: [EMPTY_SPACE, player, player, EMPTY_SPACE])
		if (line[0] == EMPTY_SPACE && line[1] == player && line[2] == player && line[3] == EMPTY_SPACE &&
			line_opp[0] == EMPTY_SPACE)
			return true;

		return false;
	}

	// For double three checking, only count one set per unique direction (e.g. indices 0–3).
	bool is_unique_direction(int index)
	{
		return (index >= 0 && index < 4);
	}
}

// returns true if placing a stone at (x,y) for 'player' creates a double-three.
bool Rules::double_three_detected(Board &board, int x, int y, int player)
{
	int opponent = (player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
	int count = 0;

	for (int i = 0; i < 8; ++i)
	{
		int dx = DIRECTIONS[i][0], dy = DIRECTIONS[i][1];
		if (!is_within_bounds(x, y, dx, dy) || !is_within_bounds(x, y, -dx, -dy))
			continue;
		if (board.get_value(x - dx, y - dy) == opponent)
			continue;
		if (check_edge(board, x, y, dx, dy, player, opponent))
		{
			++count;
			continue;
		}
		if (is_unique_direction(i))
		{
			if (check_middle(board, x, y, dx, dy, player, opponent))
				++count;
		}
	}
	return (count >= 2);
}

unsigned int extract_line_as_bits(Board &board, int x, int y, int dx, int dy, int length)
{
	unsigned int pattern = 0;
	// Loop from 1 to 'length'
	for (int i = 1; i <= length; ++i)
	{
		// Update coordinates incrementally.
		x += dx;
		y += dy;
		// Check if within bounds.
		if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE)
			return 0xFFFFFFFF;				  // Special out-of-bounds indicator.
		int cell = board.get_value_bit(x, y); // Returns 0, 1, or 2.
		// Pack the cell value into the pattern (using 2 bits per cell).
		pattern = (pattern << 2) | (cell & 0x3);
		// cell & 0x3 ensures that only the lower 2 bits of 'cell' are kept.
		// The mask 0x3 is binary 11 (i.e., 0b11), so any value in 'cell' will be reduced to its
		// two least-significant bits, effectively restricting the result to one of four possible values (0-3).
		// In our usage, we expect cell values to be 0 (empty), 1 (PLAYER_1), or 2 (PLAYER_2).
		//
		// For example, if cell = 5 (binary 101):
		//      101 (binary for 5)
		//   &  011 (binary for 0x3)
		//   ---------
		//      001 (binary for 1)
		// Thus, 5 & 0x3 yields 1, ensuring that any extraneous higher bits are ignored.
	}
	return pattern;
}

#include <iostream>

// Helper function: given a packed pattern (2 bits per cell) and the length,
// prints the line in a human-readable format.
void print_line_pattern(unsigned int pattern, int length)
{
	if (pattern == 0xFFFFFFFF)
	{
		std::cout << "Out-of-bounds" << std::endl;
		return;
	}
	// For each cell, extract the corresponding 2 bits.
	std::cout << "pattern: [";

	for (int i = 0; i < length; ++i)
	{
		// Calculate shift amount: the first cell is in the high bits.
		int shift = 2 * (length - i - 1);
		int cell = (pattern >> shift) & 0x3;
		char symbol;
		switch (cell)
		{
		case 0:
			symbol = '.';
			break;
		case 1:
			symbol = '1';
			break;
		case 2:
			symbol = '2';
			break;
		default:
			symbol = '?';
			break;
		}
		std::cout << symbol << " ";
	}
	std::cout << "]" << std::endl;
}

const unsigned int cond_1a = (PLAYER_1 << 6) | (PLAYER_1 << 4) | (EMPTY_SPACE << 2) | PLAYER_2; // 01 00 10, if PLAYER_1=1 and PLAYER_2=2.
const unsigned int cond_1a_opp = (EMPTY_SPACE << 2) | PLAYER_2;									// 01 00 10, if PLAYER_1=1 and PLAYER_2=2.

bool check_edge_bit(Board &board, int x, int y, int dx, int dy, int player, int opponent)
{
	unsigned int forward = extract_line_as_bits(board, x, y, dx, dy, 4);
	unsigned int backward = extract_line_as_bits(board, x, y, -dx, -dy, 2);
	if (forward == 0xFFFFFFFF || backward == 0xFFFFFFFF)
		return false; // out-of-bounds
	(void)player;
	(void)opponent;
	return true;
}

// returns true if placing a stone at (x,y) for 'player' creates a double-three (bitwise).
bool Rules::double_three_detected_bit(Board &board, int x, int y, int player)
{
	int opponent = (player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
	int count = 0;
	for (int i = 0; i < 8; ++i)
	{
		int dx = DIRECTIONS[i][0], dy = DIRECTIONS[i][1];
		if (!is_within_bounds(x, y, dx, dy) || !is_within_bounds(x, y, -dx, -dy))
			continue;
		if (board.get_value_bit(x - dx, y - dy) == opponent)
			continue;

		unsigned int forward = extract_line_as_bits(board, x, y, dx, dy, 3);
		unsigned int backward = extract_line_as_bits(board, x, y, -dx, -dy, 3);
		if (forward == 0xFFFFFFFF || backward == 0xFFFFFFFF)
			continue; // out-of-bounds

		std::cout << "Forward:" << std::endl;
		print_line_pattern(forward, 3);
		std::cout << "Backward:" << std::endl;
		print_line_pattern(backward, 3);
	}
	return (count >= 2);
}
