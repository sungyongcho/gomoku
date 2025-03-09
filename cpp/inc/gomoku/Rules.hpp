#ifndef RULES_HPP
#define RULES_HPP

#include "Board.hpp"
#include <vector>

class Rules
{
public:
	static std::vector<std::pair<int, int> > capture_opponent(Board &board, int x, int y, int player);

	static bool get_captured_stones(Board &board, int x, int y, const std::string &last_player,
									std::vector<std::pair<int, int> > &captured);

	static bool double_three_detected(Board &board, int x, int y, int player);

	static bool get_captured_stones_bit(Board &board, int x, int y, const std::string &last_player,
										std::vector<std::pair<int, int> > &captured);

	static bool double_three_detected_bit(Board &board, int x, int y, int player);
};

#endif
