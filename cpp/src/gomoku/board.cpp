#include "board.hpp"
#include <sstream>
#include <algorithm>

Board::Board(const std::vector<std::vector<char> >& board_data,
             int goal, const std::string& last_stone, const std::string& next_stone,
             int last_score, int next_score)
    : goal(goal), last_player(last_stone == "X" ? PLAYER_1 : PLAYER_2),
      next_player(next_stone == "X" ? PLAYER_1 : PLAYER_2),
      last_player_score(last_score), next_player_score(next_score)
{
    // Initialize board from character matrix to integer matrix
    int rows = board_data.size();
    int cols = board_data[0].size();
    position.resize(rows, std::vector<int>(cols, EMPTY_SPACE));

    for (size_t r = 0; r < board_data.size(); ++r) {
        for (size_t c = 0; c < board_data[r].size(); ++c) {
            if (board_data[r][c] == 'X')
                position[r][c] = PLAYER_1;
            else if (board_data[r][c] == 'O')
                position[r][c] = PLAYER_2;
            else
                position[r][c] = EMPTY_SPACE;
        }
    }
}

// Get value at specific position
int Board::get_value(int col, int row) const {
    return position[row][col];
}

// Set value at specific position
void Board::set_value(int col, int row, int value) {
    position[row][col] = value;
}

// Reset the board
void Board::reset_board() {
    for (size_t r = 0; r < position.size(); ++r) {
        std::fill(position[r].begin(), position[r].end(), EMPTY_SPACE);
    }
}

// Get a row from the board
std::vector<int> Board::get_row(int row) const {
    return position[row];
}

// Get a column from the board
std::vector<int> Board::get_column(int col) const {
    std::vector<int> column(position.size());
    for (size_t r = 0; r < position.size(); ++r) {
        column[r] = position[r][col];
    }
    return column;
}

// Get all downward diagonals (\)
std::vector<std::vector<int> > Board::get_all_downward_diagonals() const {
    std::vector<std::vector<int> > diagonals;
    for (int i = -NUM_LINES + 1; i < NUM_LINES; ++i) {
        std::vector<int> diag;
        for (int row = 0; row < (int)position.size(); ++row) {
            int col = row + i;
            if (col >= 0 && col < (int)position[row].size()) {
                diag.push_back(position[row][col]);
            }
        }
        diagonals.push_back(diag);
    }
    return diagonals;
}

// Get all upward diagonals (/)
std::vector<std::vector<int> > Board::get_all_upward_diagonals() const {
    std::vector<std::vector<int> > diagonals;
    for (int i = -NUM_LINES + 1; i < NUM_LINES; ++i) {
        std::vector<int> diag;
        for (int row = 0; row < (int)position.size(); ++row) {
            int col = (int)position[row].size() - 1 - row + i;
            if (col >= 0 && col < (int)position[row].size()) {
                diag.push_back(position[row][col]);
            }
        }
        diagonals.push_back(diag);
    }
    return diagonals;
}

// Update captured stones by setting them to EMPTY_SPACE
void Board::update_captured_stone(const std::vector<std::pair<int, int> >& captured_stones) {
    for (size_t i = 0; i < captured_stones.size(); ++i) {
        int x = captured_stones[i].first;
        int y = captured_stones[i].second;
        position[y][x] = EMPTY_SPACE;
    }
}

// Convert board to a string for printing
std::string Board::convert_board_for_print() const {
    std::ostringstream oss;
    for (size_t r = 0; r < position.size(); ++r) {
        for (size_t c = 0; c < position[r].size(); ++c) {
            oss << position[r][c] << " ";
        }
        oss << "\n";
    }
    return oss.str();
}
