#include "Board.hpp"

Board::Board(const std::vector<std::vector<char> >& board_data,
             int goal, const std::string& last_stone, const std::string& next_stone,
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
    for (size_t r = 0; r < board_data.size(); ++r) {
        for (size_t c = 0; c < board_data[r].size(); ++c) {
            if (board_data[r][c] == 'X')
                position[index(c, r)] = PLAYER_1;
            else if (board_data[r][c] == 'O')
                position[index(c, r)] = PLAYER_2;
            else
                position[index(c, r)] = EMPTY_SPACE;
        }
    }
}

int Board::get_value(int col, int row) const {
    return position[index(col, row)];
}

void Board::set_value(int col, int row, int value) {
    position[index(col, row)] = value;
    mark_cache_dirty();
}

void Board::reset_board() {
    std::fill(position.begin(), position.end(), EMPTY_SPACE);
    mark_cache_dirty();
}

std::string Board::convert_board_for_print() const {
    std::ostringstream oss;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int val = position[index(c, r)];
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

void Board::mark_cache_dirty() {
    cache_valid = false;
}

void Board::compute_diagonals() {
    cached_downward_diagonals.clear();
    cached_upward_diagonals.clear();

    // Compute downward diagonals (\)
    for (int i = -BOARD_SIZE + 1; i < BOARD_SIZE; ++i) {
        std::vector<int> diag;
        diag.reserve(BOARD_SIZE);
        for (int row = 0; row < BOARD_SIZE; ++row) {
            int col = row + i;
            if (col >= 0 && col < BOARD_SIZE)
                diag.push_back(position[index(col, row)]);
        }
        cached_downward_diagonals.push_back(diag);
    }

    // Compute upward diagonals (/)
    for (int i = -BOARD_SIZE + 1; i < BOARD_SIZE; ++i) {
        std::vector<int> diag;
        diag.reserve(BOARD_SIZE);
        for (int row = 0; row < BOARD_SIZE; ++row) {
            int col = BOARD_SIZE - 1 - row + i;
            if (col >= 0 && col < BOARD_SIZE)
                diag.push_back(position[index(col, row)]);
        }
        cached_upward_diagonals.push_back(diag);
    }
    cache_valid = true;
}

const std::vector<std::vector<int> >& Board::get_all_downward_diagonals() {
    if (!cache_valid) {
        compute_diagonals();
    }
    return cached_downward_diagonals;
}

const std::vector<std::vector<int> >& Board::get_all_upward_diagonals() {
    if (!cache_valid) {
        compute_diagonals();
    }
    return cached_upward_diagonals;
}
