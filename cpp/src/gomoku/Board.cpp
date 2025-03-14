#include "Board.hpp"

Board::Board(const std::vector<std::vector<char> > &board_data,
			 int goal, const std::string &last_stone, const std::string &next_stone,
			 int last_score, int next_score)
	: goal(goal),
	  last_player(last_stone == "X" ? PLAYER_1 : PLAYER_2),
	  next_player(next_stone == "X" ? PLAYER_1 : PLAYER_2),
	  last_player_score(last_score),
	  next_player_score(next_score)
{
	this->resetBitboard();
	this->initBitboardFromData(board_data);
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

void Board::initBitboardFromData(const std::vector<std::vector<char> > &board_data)
{
	for (size_t r = 0; r < board_data.size(); ++r)
	{
		for (size_t c = 0; c < board_data[r].size(); ++c)
		{
			int idx = getIndex(c, r);
			if (idx < 0)
				continue;
			if (board_data[r][c] == PLAYER_X)
			{
				setValueBit(c, r, PLAYER_1);
			}
			else if (board_data[r][c] == PLAYER_O)
			{
				setValueBit(c, r, PLAYER_2);
			}
		}
	}
}

void Board::resetBitboard()
{
	// Using memset (make sure ARRAY_SIZE * sizeof(uint64_t) is used)
	memset(this->last_player_board, 0, ARRAY_SIZE * sizeof(uint64_t));
	memset(this->next_player_board, 0, ARRAY_SIZE * sizeof(uint64_t));
}

uint64_t *Board::getBitboardByPlayer(int player)
{
	if (player == PLAYER_1)
		return this->last_player_board;
	else if (player == PLAYER_2)
		return this->next_player_board;
	throw std::invalid_argument("Invalid player value");
}

// Set the cell (col, row) to a given player.
void Board::setValueBit(int col, int row, int player)
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
inline int Board::getValueBit(int col, int row) const
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

int Board::getNextPlayer()
{
	return this->next_player;
}

int Board::getLastPlayer()
{
	return this->last_player;
}

void Board::printBitboard() const
{
	for (int r = 0; r < BOARD_SIZE; r++)
	{
		for (int c = 0; c < BOARD_SIZE; c++)
		{
			int v = getValueBit(c, r);
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

void Board::BitboardToJsonBoardboard(rapidjson::Value &json_board, rapidjson::Document::AllocatorType &allocator) const
{
	json_board.SetArray(); // Ensure it's an array type

	for (int r = 0; r < BOARD_SIZE; ++r)
	{
		rapidjson::Value json_row(rapidjson::kArrayType);
		for (int c = 0; c < BOARD_SIZE; ++c)
		{
			rapidjson::Value cell;
			char temp_str[2];
			int value = getValueBit(c, r);
			if (value == PLAYER_1)
				temp_str[0] = PLAYER_X; // 'X'
			else if (value == PLAYER_2)
				temp_str[0] = PLAYER_O; // 'O'
			else
				temp_str[0] = '.'; // empty
			temp_str[1] = '\0';
			cell.SetString(temp_str, allocator);
			json_row.PushBack(cell, allocator);
		}
		json_board.PushBack(json_row, allocator);
	}
}

std::pair<int, int> Board::getCurrentScore()
{
	std::pair<int, int> ret;

	ret.first = this->last_player_score;
	ret.second = this->next_player_score;
	return ret;
}
