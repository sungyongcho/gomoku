#ifndef MINMAX_HPP
#define MINMAX_HPP

#include "Board.hpp"
#include "Gomoku.hpp"
#include <vector>
#include <string>

#define COMPLETE_LINE_5 10000000
#define COMPLETE_LINE_4 100000
#define COMPLETE_LINE_3 1000
#define COMPLETE_LINE_2 100
#define COMPLETE_LINE_1 10
#define BLOCK_LINE_5 99900
#define BLOCK_LINE_4 10000
#define BLOCK_LINE_3 100
#define BLOCK_LINE_2 10
#define BLOCK_LINE_1 1

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

	static int evaluatiePosition(Board *&board, int player, int x, int y);
};

#endif // MINMAX_HPP
