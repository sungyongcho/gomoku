#ifndef RULES_HPP
#define RULES_HPP

#include "Board.hpp"
#include <vector>

class Rules
{
public:
	static bool getCapturedStones(Board &board, int x, int y, const std::string &last_player,
										std::vector<std::pair<int, int> > &captured);

	static bool detectDoublethreeBit(Board &board, int x, int y, int player);
};

#endif // RULES_HPP
