
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

// Recursive function to check capture condition
bool dfs_capture(Board &board, int x, int y, int player, int dx, int dy, int count)
{
	int nx = x + dx;
	int ny = y + dy;

	if (count == 3)
	{
		return board.get_value(nx, ny) == player;
	}

	int opponent = (player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
	if (board.get_value(nx, ny) != opponent)
	{
		return false;
	}

	return dfs_capture(board, nx, ny, player, dx, dy, count + 1);
}

// Check for and capture opponent stones
std::vector<std::pair<int, int> > Rules::capture_opponent(Board &board, int x, int y, int player)
{
	std::vector<std::pair<int, int> > captured_stones;

	// Use a traditional for loop instead of range-based for
	for (size_t i = 0; i < 8; ++i)
	{
		int nx = x + DIRECTIONS[i][0] * 3;
		int ny = y + DIRECTIONS[i][1] * 3;
		std::cout << nx << ", " << ny << "board:" << board.get_value(nx, ny) << std::endl;

		if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE)
		{
			continue;
		}

		if (dfs_capture(board, x, y, player, DIRECTIONS[i][0], DIRECTIONS[i][1], 1))
		{
			captured_stones.push_back(std::make_pair(x + DIRECTIONS[i][0], y + DIRECTIONS[i][1]));
			captured_stones.push_back(std::make_pair(x + DIRECTIONS[i][0] * 2, y + DIRECTIONS[i][1] * 2));

			board.set_value(x + DIRECTIONS[i][0], y + DIRECTIONS[i][1], EMPTY_SPACE);
			board.set_value(x + DIRECTIONS[i][0] * 2, y + DIRECTIONS[i][1] * 2, EMPTY_SPACE);
		}
	}

	return captured_stones;
}

void Rules::remove_captured_stone(Board &board, std::vector<std::pair<int, int> > &captured_stones)
{
	std::vector<std::pair<int, int> >::iterator it;

	for (it = captured_stones.begin(); it != captured_stones.end(); it++)
	{
		std::cout << it->first << "," << it->second << std::endl;

		board.set_value(it->first, it->second, EMPTY_SPACE);
		// std::cout << it->first << std::endl;
	}
}

// New helper that fills 'captured' with captured stones and returns whether any were found.
bool Rules::get_captured_stones(Board &board, int x, int y, const std::string &last_player,
                                  std::vector<std::pair<int, int> >& captured)
{
    int currentPlayer = (last_player == "X") ? PLAYER_1 : PLAYER_2;
    captured = Rules::capture_opponent(board, x, y, currentPlayer);
    return !captured.empty();
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
// New method: returns true if placing a stone at (x,y) for 'player' creates a double-three.
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
