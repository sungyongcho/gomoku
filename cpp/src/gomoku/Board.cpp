#include "Board.hpp"

Board::Board(const std::vector<std::vector<char> > &board_data,
			 int goal, const std::string &last_stone, const std::string &next_stone,
			 int last_score, int next_score)
	: goal(goal),
	  last_player(last_stone == "X" ? PLAYER_1 : PLAYER_2),
	  next_player(next_stone == "X" ? PLAYER_1 : PLAYER_2),
	  last_player_score(last_score),
	  next_player_score(next_score),
	  position(BOARD_SIZE * BOARD_SIZE, EMPTY_SPACE),
	  cache_valid(false)
{
	// Initialize board (convert each char to an integer value)
	for (size_t r = 0; r < board_data.size(); ++r)
	{
		for (size_t c = 0; c < board_data[r].size(); ++c)
		{
			if (board_data[r][c] == this->last_player)
			{
				position[getIndex(c, r)] = PLAYER_1;
				set_board(c, r, true);
			}
			else if (board_data[r][c] == this->next_player)
			{
				position[getIndex(c, r)] = PLAYER_2;
				set_board(c, r, false);
			}
			else
				position[getIndex(c, r)] = EMPTY_SPACE;
		}
	}
}

int Board::getIndex(int col, int row) const
{
	if (col < 0 || col >= BOARD_SIZE || row < 0 || row >= BOARD_SIZE)
	{
		std::cout << "Error: index wrong" << std::endl;
		return -1;
	}
	return row * BOARD_SIZE + col;
}

// bitmask

uint64_t* Board::get_bitboard_by_player(int player)
{
    if (player == PLAYER_1)
        return this->last_player_board;
    else if (player == PLAYER_2)
        return this->next_player_board;

    throw std::invalid_argument("Invalid player value");
}


int Board::get(uint64_t (&player_board)[ARRAY_SIZE], int col, int row)
{
	// Calculate index directly
	int index = getIndex(col, row);
	if (index < 0)
		return -1;

	// Calculate array index and bit position
	int arrayIndex = index / UINT64_BITS;
	int bitPos = index % UINT64_BITS;

	return (player_board[arrayIndex] >> bitPos) & 1;
}

void Board::set_board(int col, int row, bool is_last)
{
	int index = getIndex(col, row);
	if (index < 0)
		return;

	int arrayIndex = index / UINT64_BITS;
	int bitPos = index % UINT64_BITS;

	if (is_last)
		this->last_player_board[arrayIndex] |= ((uint64_t)1 << bitPos);
	else
		this->next_player_board[arrayIndex] |= ((uint64_t)1 << bitPos);
}

// Set the cell (col, row) to a given player.
inline void Board::set_value_bit(int col, int row, int player)
{
	int idx = getIndex(col, row);
	if (idx < 0)
		return;
	int word = idx / UINT64_BITS;
	int bit = idx % UINT64_BITS;
	uint64_t mask = (uint64_t)1 << bit;
	// Clear cell in both boards.
	this->last_player_board[word] &= ~mask;
	this->next_player_board[word] &= ~mask;
	if (player == PLAYER_1)
		this->last_player_board[word] |= mask;
	else if (player == PLAYER_2)
		this->next_player_board[word] |= mask;
}

// Get the value at (col, row).
inline int Board::get_value_bit(int col, int row) const
{
	int idx = getIndex(col, row);
	if (idx < 0)
		return EMPTY_SPACE;
	int word = idx / UINT64_BITS;
	int bit = idx % UINT64_BITS;
	uint64_t mask = (uint64_t)1 << bit;
	if (this->last_player_board[word] & mask)
		return PLAYER_1;
	if (this->next_player_board[word] & mask)
		return PLAYER_2;
	return EMPTY_SPACE;
}
void Board::print_board_bit() const
{
	for (int r = 0; r < BOARD_SIZE; r++)
	{
		for (int c = 0; c < BOARD_SIZE; c++)
		{
			int v = get_value(c, r);
			if (v == PLAYER_1)
				std::cout << "X ";
			else if (v == PLAYER_2)
				std::cout << "O ";
			else
				std::cout << ". ";
		}
		std::cout << std::endl;
	}
}

// bitmask

int Board::get_value(int col, int row) const
{
	return position[getIndex(col, row)];
}

void Board::set_value(int col, int row, int value)
{
	position[getIndex(col, row)] = value;
	mark_cache_dirty();
}

void Board::reset_board()
{
	std::fill(position.begin(), position.end(), EMPTY_SPACE);
	mark_cache_dirty();
}

std::string Board::convert_board_for_print() const
{
	std::ostringstream oss;
	for (int r = 0; r < BOARD_SIZE; ++r)
	{
		for (int c = 0; c < BOARD_SIZE; ++c)
		{
			int val = position[getIndex(c, r)];
			if (val == PLAYER_1)
				oss << "X ";
			else if (val == PLAYER_2)
				oss << "O ";
			else
				oss << ". ";
		}
		oss << "\n";
	}
	return oss.str();
}

void Board::mark_cache_dirty()
{
	cache_valid = false;
}

void Board::compute_diagonals()
{
	cached_downward_diagonals.clear();
	cached_upward_diagonals.clear();

	// Compute downward diagonals (\)
	for (int i = -BOARD_SIZE + 1; i < BOARD_SIZE; ++i)
	{
		std::vector<int> diag;
		diag.reserve(BOARD_SIZE);
		for (int row = 0; row < BOARD_SIZE; ++row)
		{
			int col = row + i;
			if (col >= 0 && col < BOARD_SIZE)
				diag.push_back(position[getIndex(col, row)]);
		}
		cached_downward_diagonals.push_back(diag);
	}

	// Compute upward diagonals (/)
	for (int i = -BOARD_SIZE + 1; i < BOARD_SIZE; ++i)
	{
		std::vector<int> diag;
		diag.reserve(BOARD_SIZE);
		for (int row = 0; row < BOARD_SIZE; ++row)
		{
			int col = BOARD_SIZE - 1 - row + i;
			if (col >= 0 && col < BOARD_SIZE)
				diag.push_back(position[getIndex(col, row)]);
		}
		cached_upward_diagonals.push_back(diag);
	}
	cache_valid = true;
}

const std::vector<std::vector<int> > &Board::get_all_downward_diagonals()
{
	if (!cache_valid)
	{
		compute_diagonals();
	}
	return cached_downward_diagonals;
}

const std::vector<std::vector<int> > &Board::get_all_upward_diagonals()
{
	if (!cache_valid)
	{
		compute_diagonals();
	}
	return cached_upward_diagonals;
}

// Convert internal vector<int> board back to 2D char array representation
std::vector<std::vector<char> > Board::to_char_board() const
{
	// Create a 2D char board initialized with '.'
	std::vector<std::vector<char> > char_board(BOARD_SIZE, std::vector<char>(BOARD_SIZE, '.'));

	// Loop through the board and convert the integer values to char
	for (int row = 0; row < BOARD_SIZE; ++row)
	{
		for (int col = 0; col < BOARD_SIZE; ++col)
		{
			int value = position[getIndex(col, row)];
			if (value == PLAYER_1)
			{
				char_board[row][col] = 'X';
			}
			else if (value == PLAYER_2)
			{
				char_board[row][col] = 'O';
			}
			else
			{
				char_board[row][col] = '.';
			}
		}
	}

	return char_board;
}

// Convert the board to a JSON array (modifies the passed rapidjson::Value reference)
void Board::to_json_board(rapidjson::Value &json_board, rapidjson::Document::AllocatorType &allocator) const
{
	json_board.SetArray(); // Ensure it's an array type

	std::vector<std::vector<char> > char_board = this->to_char_board();
	for (int i = 0; i < BOARD_SIZE; ++i)
	{
		rapidjson::Value json_row(rapidjson::kArrayType);
		for (int j = 0; j < BOARD_SIZE; ++j)
		{
			rapidjson::Value cell;
			char temp_str[2] = {char_board[i][j], '\0'}; // Char to string
			cell.SetString(temp_str, allocator);
			json_row.PushBack(cell, allocator);
		}
		json_board.PushBack(json_row, allocator);
	}
}

std::pair<int, int> Board::get_current_score()
{
	std::pair<int, int> ret;

	ret.first = this->last_player_score;
	ret.second = this->next_player_score;
	return ret;
}
