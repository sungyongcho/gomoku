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

// Define constants
#define PLAYER_1 1
#define PLAYER_2 2
#define EMPTY_SPACE 0
#define BOARD_SIZE 19 // Standard Gomoku board size

// for bitwise
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE) // 361 cells
#define UINT64_BITS 64
#define ARRAY_SIZE ((TOTAL_CELLS + UINT64_BITS - 1) / UINT64_BITS) // 6 elements


class Board
{
public:
	// Constructor: initialize the board from a 2D char array (e.g. from JSON)
	Board(const std::vector<std::vector<char> > &board_data,
		  int goal, const std::string &last_stone, const std::string &next_stone,
		  int last_score, int next_score);

	// Accessors and mutators
	int get_value(int col, int row) const;
	void set_value(int col, int row, int value);
	void reset_board();
	std::pair<int, int> get_current_score();
	// Convert internal vector<int> board back to 2D char array representation
	std::vector<std::vector<char> > to_char_board() const;
	void to_json_board(rapidjson::Value &json_board, rapidjson::Document::AllocatorType& allocator) const;

	// These functions now use caching. They return a reference to the cached vector.
	const std::vector<std::vector<int> > &get_all_downward_diagonals();
	const std::vector<std::vector<int> > &get_all_upward_diagonals();

	// Other utility functions
	std::string convert_board_for_print() const;

	// When the board changes, call this to mark the cache as dirty.
	void mark_cache_dirty();

	uint64_t *get_bitboard_by_player(int player);
	int getIndex(int col, int row) const;
	inline void set_value_bit(int col, int row, int player);
	inline int get_value_bit(int col, int row) const;
	void print_board_bit() const;


private:
	int goal;
	int last_player;
	int next_player;
	int last_player_score;
	int next_player_score;

	uint64_t last_player_board[ARRAY_SIZE];
	uint64_t next_player_board[ARRAY_SIZE];


	int get(uint64_t (&player_board)[ARRAY_SIZE], int col, int row);
	void set_board(int col, int row, bool is_last);


	// Store board as a 1D vector for speed.
	std::vector<int> position;

	// Cached diagonals
	std::vector<std::vector<int> > cached_downward_diagonals;
	std::vector<std::vector<int> > cached_upward_diagonals;
	bool cache_valid;

	// Convert 2D coordinates into a 1D index.
	inline int index(int col, int row) const { return row * BOARD_SIZE + col; }

	// Recompute diagonals and update cache.
	void compute_diagonals();
};

#endif
