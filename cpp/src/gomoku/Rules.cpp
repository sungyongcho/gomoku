#include "Rules.hpp"
#include <iostream>

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
	bool is_unique_direction_removeit(int index)
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
		if (is_unique_direction_removeit(i))
		{
			if (check_middle(board, x, y, dx, dy, player, opponent))
				++count;
		}
	}
	return (count >= 2);
}

/**
 * make sure to understand the function stores then shift the function stored will be shifted to LEFT
 */
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

// Helper function: given a packed pattern (2 bits per cell) and the length,
// prints the line in a human-readable format.
void print_line_pattern_reverse(unsigned int pattern, int length)
{
	if (pattern == 0xFFFFFFFF)
	{
		std::cout << "Out-of-bounds" << std::endl;
		return;
	}
	std::cout << "pattern (reversed): [";

	for (int i = 0; i < length; ++i)
	{
		// Now, cell0 comes from the lowest bits.
		int shift = 2 * i;
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

// Helper functions to pack cell values into an unsigned int.
// Each cell uses 2 bits.

bool is_unique_direction(int index)
{
	return (index >= 0 && index < 4);
}

inline unsigned int pack_cells_4(unsigned int a, unsigned int b, unsigned int c, unsigned int d)
{
	return (a << 6) | (b << 4) | (c << 2) | d;
}

inline unsigned int pack_cells_3(unsigned int a, unsigned int b, unsigned int c)
{
	return (a << 6) | (b << 4) | (c << 2);
}

inline unsigned int pack_cells_2(unsigned int a, unsigned int b)
{
	return (a << 6) | (b << 4);
}

inline unsigned int pack_cells_1(unsigned int a)
{
	return (a << 6); // a occupies 2 bits.
}

bool check_edge_bit_case_1(unsigned int forward, unsigned int backward, int player, int opponent)
{
	unsigned int cond_1a_fwd = pack_cells_4(player, player, EMPTY_SPACE, opponent);
	unsigned int cond_1a_bkwd = pack_cells_2(EMPTY_SPACE, opponent);

	unsigned int cond_1b_fwd = pack_cells_4(player, player, EMPTY_SPACE, player);

	// Fallback expected pattern using only the first 3 forward cells
	// and the first backward cell.
	unsigned int cond1_fwd = pack_cells_3(player, player, EMPTY_SPACE);
	unsigned int cond1_bkwd = pack_cells_1(EMPTY_SPACE);

	unsigned int forward_three_cells = forward & 0xFC;
	unsigned int backward_one_cell = backward & 0xC0;

	bool cond_1a = (cond_1a_fwd == forward) && (cond_1a_bkwd == backward);
	bool cond_1b = (cond_1b_fwd == forward);

	if (!(cond_1a || cond_1b) && (cond1_fwd == forward_three_cells) && (cond1_bkwd == backward_one_cell))
		return true;

	return false;
}

bool check_edge_bit_case_2(unsigned int forward, unsigned int backward, int player)
{
	unsigned int cond_2a_fwd = pack_cells_3(player, EMPTY_SPACE, player);
	unsigned int cond_2a_bkwd = pack_cells_2(EMPTY_SPACE, player);

	unsigned int cond2_fwd = pack_cells_4(player, EMPTY_SPACE, player, EMPTY_SPACE);
	unsigned int cond2_bkwd = pack_cells_1(EMPTY_SPACE);

	unsigned int forward_three_cells = forward & 0xFC;
	unsigned int backward_one_cell = backward & 0xC0;

	bool cond_2a = (cond_2a_fwd == forward_three_cells) && (cond_2a_bkwd == backward);

	if (!(cond_2a) && (cond2_fwd == forward) && (cond2_bkwd == backward_one_cell))
		return true;

	return false;
}

bool check_edge_bit_case_3(unsigned int forward, unsigned int backward, int player)
{
	unsigned int cond3_fwd = pack_cells_4(EMPTY_SPACE, player, player, EMPTY_SPACE);
	unsigned int cond3_bkwd = pack_cells_1(EMPTY_SPACE);

	unsigned int backward_one_cell = backward & 0xC0;

	if ((cond3_fwd == forward) && (cond3_bkwd == backward_one_cell))
		return true;

	return false;
}

bool check_edge_bit(Board &board, int x, int y, int dx, int dy, int player, int opponent)
{
	unsigned int forward = extract_line_as_bits(board, x, y, dx, dy, 4);
	unsigned int backward = extract_line_as_bits(board, x, y, -dx, -dy, 2);

	// because extracting only 2 cells so shifting 2 cells left to align
	backward = (backward << 4) & 0xF0;

	if (forward == 0xFFFFFFFF || backward == 0xFFFFFFFF)
		return false; // out-of-bounds

	if (check_edge_bit_case_1(forward, backward, player, opponent))
		return true;

	if (check_edge_bit_case_2(forward, backward, player))
		return true;

	if (check_edge_bit_case_3(forward, backward, player))
		return true;

	return false;
}

bool check_middle_bit_case_1(unsigned int forward, unsigned int backward, int player, int opponent)
{
	unsigned int cond1a_fwd = pack_cells_3(player, EMPTY_SPACE, opponent);
	unsigned int cond1a_bkwd = pack_cells_3(player, EMPTY_SPACE, opponent);

	unsigned int cond1b_fwd = pack_cells_3(player, EMPTY_SPACE, player);
	unsigned int cond1b_bkwd = pack_cells_1(player);

	unsigned int cond1c_fwd = pack_cells_2(player, EMPTY_SPACE);
	unsigned int cond1c_bkwd = pack_cells_2(player, EMPTY_SPACE);

	unsigned int forward_three_cells = forward & 0xFC;
	unsigned int backward_three_cells = backward & 0xFC;

	unsigned int forward_two_cells = forward & 0xF0;
	unsigned int backward_two_cells = backward & 0xF0;

	unsigned int backward_one_cell = backward & 0xC0;

	unsigned int cond1a = (cond1a_fwd == forward_three_cells) && (cond1a_bkwd == backward_three_cells);
	unsigned int cond1b = (cond1b_fwd == forward_three_cells) && (cond1b_bkwd == backward_one_cell);
	unsigned int cond1c = (cond1c_fwd == forward_two_cells) && (cond1c_bkwd == backward_two_cells);

	// std::cout << "----fwd------" << std::endl;
	// print_line_pattern_reverse(cond1c_fwd, 4);
	// print_line_pattern_reverse(forward_three_cells, 4);
	// std::cout << "----fwd-----" << std::endl;
	// std::cout << "----bwd-----" << std::endl;
	// print_line_pattern_reverse(cond1c_bkwd, 4);
	// print_line_pattern_reverse(backward, 4);
	// std::cout << "----bwd-----" << std::endl;
	if (!(cond1a || cond1b) && cond1c)
		return true;

	return false;
}

bool check_middle_bit_case_2(unsigned int forward, unsigned int backward, int player)
{
	unsigned int cond2_fwd = pack_cells_3(EMPTY_SPACE, player, EMPTY_SPACE);
	unsigned int cond2_bkwd = pack_cells_2(player, EMPTY_SPACE);

	unsigned int backward_two_cells = (backward >> 4) & 0x0F;
	unsigned int forward_three_cells = (forward >> 2) & 0x3F;

	if ((cond2_fwd == forward_three_cells) && (cond2_bkwd == backward_two_cells))
		return true;

	return false;
}

bool check_middle_bit(Board &board, int x, int y, int dx, int dy, int player, int opponent)
{
	unsigned int forward = extract_line_as_bits(board, x, y, dx, dy, 3);
	unsigned int backward = extract_line_as_bits(board, x, y, -dx, -dy, 3);

	forward = (forward << 2) & 0xFC;
	backward = (backward << 2) & 0xFC;

	std::cout << "----fwd------" << std::endl;
	print_line_pattern(forward, 4);
	std::cout << "----fwd-----" << std::endl;
	std::cout << "----bwd-----" << std::endl;
	print_line_pattern(backward, 4);
	std::cout << "----bwd-----" << std::endl;

	if (forward == 0xFFFFFFFF || backward == 0xFFFFFFFF)
		return false; // out-of-bounds

	if (check_middle_bit_case_1(forward, backward, player, opponent))
		return true;

	if (check_middle_bit_case_2(forward, backward, player))
		return true;

	return false;
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
		if (check_edge_bit(board, x, y, dx, dy, player, opponent))
		{
			++count;
			std::cout << "edge count:" << count << std::endl;
			continue;
		}
		if (is_unique_direction(i))
		{
			if (check_middle_bit(board, x, y, dx, dy, player, opponent))
			{
				++count;
				std::cout << "middle count:" << count << std::endl;
			}
		}
	}
	return (count >= 2);
}
