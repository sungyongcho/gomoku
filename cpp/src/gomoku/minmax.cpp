#include "minmax.hpp"
#include <sstream>
#include <limits>
#include <cstdlib>

// //---------------------------------------------------------------------------
// // Internal types and helper functions for evaluation
// //---------------------------------------------------------------------------

// struct PatternScore
// {
// 	std::string pattern; // e.g. "XXXXX", "-XXXX-", etc.
// 	long long score;	 // Score for each occurrence.
// };

// // Define pattern table for black (PLAYER_1) evaluation.
// // For white, roles will be swapped.
// static const PatternScore patternTable[] = {
// 	{"XXXXX", 10000000000LL},
// 	{"-XXXX-", 1000000LL},
// 	{"XXXX-", 100000LL},
// 	{"-XXXX", 100000LL},
// 	{"-XXX-", 10000LL},
// 	{"XXX-", 1000LL},
// 	{"-XXX", 1000LL},
// 	{"-XX-", 100LL},
// 	{"XX-", 50LL},
// 	{"-XX", 50LL}};

// static inline char cellToChar(int cell)
// {
// 	return (cell == PLAYER_1) ? 'X' : (cell == PLAYER_2) ? 'O'
// 														 : '-';
// }

// // Extract all board lines (rows, columns, and diagonals) as strings.
// static void getAllLines(const Board &board, std::vector<std::string> &lines)
// {
// 	const int size = BOARD_SIZE;
// 	std::string line;

// 	// Horizontal lines.
// 	for (int row = 0; row < size; ++row)
// 	{
// 		line = "";
// 		for (int col = 0; col < size; ++col)
// 		{
// 			line.push_back(cellToChar(board.getValueBit(col, row)));
// 		}
// 		lines.push_back(line);
// 	}

// 	// Vertical lines.
// 	for (int col = 0; col < size; ++col)
// 	{
// 		line = "";
// 		for (int row = 0; row < size; ++row)
// 		{
// 			line.push_back(cellToChar(board.getValueBit(col, row)));
// 		}
// 		lines.push_back(line);
// 	}

// 	// Diagonals: top-left to bottom-right.
// 	for (int startCol = 0; startCol < size; ++startCol)
// 	{
// 		line = "";
// 		for (int row = 0, col = startCol; row < size && col < size; ++row, ++col)
// 		{
// 			line.push_back(cellToChar(board.getValueBit(col, row)));
// 		}
// 		lines.push_back(line);
// 	}
// 	for (int startRow = 1; startRow < size; ++startRow)
// 	{
// 		line = "";
// 		for (int row = startRow, col = 0; row < size && col < size; ++row, ++col)
// 		{
// 			line.push_back(cellToChar(board.getValueBit(col, row)));
// 		}
// 		lines.push_back(line);
// 	}

// 	// Diagonals: top-right to bottom-left.
// 	for (int startCol = size - 1; startCol >= 0; --startCol)
// 	{
// 		line = "";
// 		for (int row = 0, col = startCol; row < size && col >= 0; ++row, --col)
// 		{
// 			line.push_back(cellToChar(board.getValueBit(col, row)));
// 		}
// 		lines.push_back(line);
// 	}
// 	for (int startRow = 1; startRow < size; ++startRow)
// 	{
// 		line = "";
// 		for (int row = startRow, col = size - 1; row < size && col >= 0; ++row, --col)
// 		{
// 			line.push_back(cellToChar(board.getValueBit(col, row)));
// 		}
// 		lines.push_back(line);
// 	}
// }

// // Count overlapping occurrences of a pattern in a line.
// static int countOccurrences(const std::string &line, const std::string &pattern)
// {
// 	int count = 0;
// 	std::string::size_type pos = line.find(pattern);
// 	while (pos != std::string::npos)
// 	{
// 		++count;
// 		pos = line.find(pattern, pos + 1);
// 	}
// 	return count;
// }

// //---------------------------------------------------------------------------
// // Minimax namespace implementations
// //---------------------------------------------------------------------------

// // Advanced evaluation using pattern matching.
// long long Minmax::evaluateBoard(const Board &board, int player)
// {
// 	// int opponent = (player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
// 	// Build pattern strings for the current player.
// 	std::vector<PatternScore> playerPatterns;
// 	std::vector<PatternScore> opponentPatterns;
// 	const int numPatterns = sizeof(patternTable) / sizeof(patternTable[0]);

// 	for (int i = 0; i < numPatterns; ++i)
// 	{
// 		PatternScore p = patternTable[i]; // for PLAYER_1 ("X")
// 		if (player == PLAYER_1)
// 		{
// 			playerPatterns.push_back(p);
// 		}
// 		else
// 		{
// 			// For PLAYER_2, swap 'X' with 'O'.
// 			for (size_t j = 0; j < p.pattern.size(); ++j)
// 			{
// 				if (p.pattern[j] == 'X')
// 					p.pattern[j] = 'O';
// 				else if (p.pattern[j] == 'O')
// 					p.pattern[j] = 'X';
// 			}
// 			playerPatterns.push_back(p);
// 		}

// 		// Opponent patterns are the inverse with negative scores.
// 		PatternScore op = p;
// 		op.score = -op.score;
// 		opponentPatterns.push_back(op);
// 	}

// 	std::vector<std::string> lines;
// 	getAllLines(board, lines);

// 	long long totalScore = 0;
// 	// Scan each line for patterns.
// 	for (size_t i = 0; i < lines.size(); ++i)
// 	{
// 		const std::string &line = lines[i];
// 		for (size_t j = 0; j < playerPatterns.size(); ++j)
// 		{
// 			int occurrences = countOccurrences(line, playerPatterns[j].pattern);
// 			if (occurrences > 0)
// 				totalScore += occurrences * playerPatterns[j].score;
// 		}
// 		for (size_t j = 0; j < opponentPatterns.size(); ++j)
// 		{
// 			int occurrences = countOccurrences(line, opponentPatterns[j].pattern);
// 			if (occurrences > 0)
// 				totalScore += occurrences * opponentPatterns[j].score;
// 		}
// 	}

// 	return totalScore;
// }

// // Generate legal moves by scanning for empty cells adjacent to an occupied cell.
// std::vector<Move> Minmax::generateLegalMoves(const Board &board)
// {
// 	std::vector<Move> moves;
// 	const int size = BOARD_SIZE;
// 	// Define offsets for all neighbors.
// 	int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
// 	int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// 	for (int row = 0; row < size; ++row)
// 	{
// 		for (int col = 0; col < size; ++col)
// 		{
// 			if (board.getValueBit(col, row) != EMPTY_SPACE)
// 				continue;

// 			bool adjacent = false;
// 			for (int k = 0; k < 8; ++k)
// 			{
// 				int nx = col + dx[k];
// 				int ny = row + dy[k];
// 				if (nx >= 0 && nx < size && ny >= 0 && ny < size)
// 				{
// 					if (board.getValueBit(nx, ny) != EMPTY_SPACE)
// 					{
// 						adjacent = true;
// 						break;
// 					}
// 				}
// 			}
// 			// If no stone is adjacent, skip this cell.
// 			if (adjacent)
// 			{
// 				Move move = {col, row};
// 				moves.push_back(move);
// 			}
// 		}
// 	}
// 	// If no move was found (e.g., empty board), add center cell.
// 	if (moves.empty())
// 	{
// 		Move center = {size / 2, size / 2};
// 		moves.push_back(center);
// 	}
// 	return moves;
// }

// // Minimax with alpha-beta pruning.
// // 'player' is the maximizer; 'maximizing' indicates current turn.
// int Minmax::minimax(Board &board, int depth, int alpha, int beta, int player, bool maximizing)
// {
// 	// At depth zero, return the evaluation.
// 	if (depth == 0)
// 	{
// 		return static_cast<int>(evaluateBoard(board, player));
// 	}

// 	std::vector<Move> moves = generateLegalMoves(board);
// 	// Determine whose turn it is.
// 	int currentPlayer = maximizing ? player : ((player == PLAYER_1) ? PLAYER_2 : PLAYER_1);

// 	if (maximizing)
// 	{
// 		int maxEval = -1000000000;
// 		for (size_t i = 0; i < moves.size(); ++i)
// 		{
// 			board.setValueBit(moves[i].x, moves[i].y, currentPlayer);
// 			int eval = minimax(board, depth - 1, alpha, beta, player, false);
// 			board.setValueBit(moves[i].x, moves[i].y, EMPTY_SPACE); // undo move
// 			if (eval > maxEval)
// 				maxEval = eval;
// 			if (maxEval > alpha)
// 				alpha = maxEval;
// 			if (beta <= alpha)
// 				break; // prune branch
// 		}
// 		return maxEval;
// 	}
// 	else
// 	{
// 		int minEval = 1000000000;
// 		for (size_t i = 0; i < moves.size(); ++i)
// 		{
// 			board.setValueBit(moves[i].x, moves[i].y, currentPlayer);
// 			int eval = minimax(board, depth - 1, alpha, beta, player, true);
// 			board.setValueBit(moves[i].x, moves[i].y, EMPTY_SPACE); // undo move
// 			if (eval < minEval)
// 				minEval = eval;
// 			if (minEval < beta)
// 				beta = minEval;
// 			if (beta <= alpha)
// 				break; // prune branch
// 		}
// 		return minEval;
// 	}
// }

// // Determine the best move for 'player' using minimax search at given depth.
// Move Minmax::getBestMove(Board &board, int player, int depth)
// {
// 	std::vector<Move> moves = generateLegalMoves(board);
// 	Move bestMove = moves[0];
// 	int bestScore = -1000000000;
// 	int alpha = -1000000000;
// 	int beta = 1000000000;

// 	for (size_t i = 0; i < moves.size(); ++i)
// 	{
// 		board.setValueBit(moves[i].x, moves[i].y, player);
// 		int score = Minmax::minimax(board, depth - 1, alpha, beta, player, false);
// 		board.setValueBit(moves[i].x, moves[i].y, EMPTY_SPACE);
// 		if (score > bestScore)
// 		{
// 			bestScore = score;
// 			bestMove = moves[i];
// 		}
// 	}
// 	return bestMove;
// }

inline unsigned int pack_cells_4(unsigned int a, unsigned int b, unsigned int c, unsigned int d)
{
	return (a << 6) | (b << 4) | (c << 2) | d;
}

inline unsigned int pack_cells_3(unsigned int a, unsigned int b, unsigned int c)
{
	return (a << 4) | (b << 2) | c;
}

inline unsigned int pack_cells_2(unsigned int a, unsigned int b)
{
	return (a << 2) | b;
}

inline unsigned int pack_cells_1(unsigned int a)
{
	return a;
}

int check_one_direction_score(unsigned int pattern, int player, int length)
{
	if (length <= 0)
		return 0;
	unsigned int and_opr;
	int opponent = OPPONENT(player);
	(void)opponent;
	and_opr = 0xFF;
	if (length == 1)
		and_opr = 0x03;
	else if (length == 2)
		and_opr = 0x0F;
	else if (length == 3)
		and_opr = 0x3F;

	pattern = (pattern >> 2 * length) & and_opr;
	if (length == 4)
	{
		if (pattern == pack_cells_4(player, player, player, player))
			return COMPLETE_LINE_4;
		else if (pattern == pack_cells_4(player, player, player, EMPTY_SPACE))
			return COMPLETE_LINE_3;
		else if (pattern == pack_cells_4(player, player, EMPTY_SPACE, EMPTY_SPACE))
			return COMPLETE_LINE_2;
		else if (pattern == pack_cells_4(player, EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE))
			return COMPLETE_LINE_1;
		else if (pattern == pack_cells_4(EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE))
			return EMPTY_SPACE;
	}
	if (length == 3)
	{
		if (pattern == pack_cells_3(player, player, player))
			return COMPLETE_LINE_3;
		else if (pattern == pack_cells_3(player, player, EMPTY_SPACE))
			return COMPLETE_LINE_2;
		else if (pattern == pack_cells_3(player, EMPTY_SPACE, EMPTY_SPACE))
			return COMPLETE_LINE_1;
		else if (pattern == pack_cells_3(EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE))
			return EMPTY_SPACE;
	}
	// and so on ...
	return 22; // just testing value no meaning
}

int Minmax::evaluatiePosition(Board *&board, int player, int x, int y)
{
	int opponent = OPPONENT(player);

	std::cout << "check: " << board->getValueBit(x, y) << std::endl;
	std::cout << "player: " << player << " opponent: " << opponent << std::endl;

	for (int i = 0; i < 4; ++i)
	{
		int fwd = i;
		// int bkwd = i + 4;
		// std::cout << "i: " << i
		// 		  << "\nfwd: " << Board::convertIndexToCoordinates(x + DIRECTIONS[fwd][0], y + DIRECTIONS[fwd][1])
		// 		  << "\nbkwd: " << Board::convertIndexToCoordinates(x + DIRECTIONS[bkwd][0], y + DIREC;
		unsigned int fwd_pattern = board->extractLineAsBits(x,y,DIRECTIONS[fwd][0], DIRECTIONS[fwd][1], 4);

		Board::printLinePattern(fwd_pattern, Board::getCellCount(fwd_pattern, 4));
		std::cout << "fwd score: " << check_one_direction_score(fwd_pattern, player, Board::getCellCount(fwd_pattern, 4));

	}

	return 0;
}
