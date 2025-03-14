#ifndef MINMAX_HPP
#define MINMAX_HPP

#include "Board.hpp"
#include <vector>
#include <string>

// A simple move structure.
struct Move
{
	int x;
	int y;
};

class Minmax
{
private:
	// Advanced board evaluation function using pattern matching.
	// Positive scores favor the given 'player'.
	static long long evaluateBoard(const Board &board, int player);

	// Generate legal moves.
	// Only returns empty cells that are adjacent to any occupied cell.
	static std::vector<Move> generateLegalMoves(const Board &board);

	// The minimax function with alpha-beta pruning.
	// 'depth' is the search depth, 'player' is the maximizer.
	// 'maximizing' indicates whose turn it is.
	static int minimax(Board &board, int depth, int alpha, int beta, int player, bool maximizing);
public:
	// Returns the best move for 'player' using minimax search at given 'depth'.
	static Move getBestMove(Board &board, int player, int depth);
};

#endif // MINMAX_HPP
