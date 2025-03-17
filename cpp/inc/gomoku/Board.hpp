#ifndef BOARD_HPP
#define BOARD_HPP

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <stdexcept>

#include "Gomoku.hpp"

// for bitwise
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE) // 361 cells
#define UINT64_BITS 64
#define ARRAY_SIZE ((TOTAL_CELLS + UINT64_BITS - 1) / UINT64_BITS) // 6 elements

class Board
{
private:
	int goal;
	int last_player;
	int next_player;
	int last_player_score;
	int next_player_score;

	uint64_t last_player_board[ARRAY_SIZE];
	uint64_t next_player_board[ARRAY_SIZE];

	void resetBitboard();
	void initBitboardFromData(const std::vector<std::vector<char> > &board_data);

public:
	Board(const std::vector<std::vector<char> > &board_data,
		  int goal, const std::string &last_stone, const std::string &next_stone,
		  int last_score, int next_score);

	std::pair<int, int> getCurrentScore();
	uint64_t *getBitboardByPlayer(int player);

	int getIndex(int col, int row) const;

	void setValueBit(int col, int row, int player);
	int getValueBit(int col, int row) const;

	static unsigned int getCellCount(unsigned int pattern);
	unsigned int extractLineAsBits(int x, int y, int dx, int dy, int length);

	int getLastPlayer();
	int getNextPlayer();

	void printBitboard() const;
	void BitboardToJsonBoardboard(rapidjson::Value &json_board, rapidjson::Document::AllocatorType &allocator) const;

	static std::string convertIndexToCoordinates(int col, int row);

	static void printLinePattern(unsigned int pattern, int length);
	static void printLinePatternReverse(unsigned int pattern, int length);
};

#endif // BOARD_HPP
