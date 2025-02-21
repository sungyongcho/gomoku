#ifndef RULES_HPP
#define RULES_HPP

#include "Board.hpp"
#include <vector>

class Rules
{
public:
	static std::vector<std::pair<int, int> > capture_opponent(Board &board, int x, int y, int player);
	static void remove_captured_stone(Board &board, std::vector<std::pair<int, int> > &captured_stones);

	static bool double_three_detected(Board &board, int x, int y, int player);

};

#endif
