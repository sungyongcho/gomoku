#ifndef RULES_HPP
#define RULES_HPP

#include "Board.hpp"
#include <vector>

class Rules {
public:
    static std::vector<std::pair<int, int> > capture_opponent(Board& board, int x, int y, int player);
    // static bool double_three_detected(const Board& board, int x, int y, int player);
};

#endif

