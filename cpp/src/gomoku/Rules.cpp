
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
