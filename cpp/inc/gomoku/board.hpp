#ifndef BOARD_HPP
#define BOARD_HPP

#include <vector>
#include <string>
#include <iostream>

#define PLAYER_1 1
#define PLAYER_2 2
#define EMPTY_SPACE 0
#define NUM_LINES 15  // Define NUM_LINES as per your board size

class Board {
public:
    Board(const std::vector<std::vector<char> >& board_data,
          int goal, const std::string& last_stone, const std::string& next_stone,
          int last_score, int next_score);

    int get_value(int col, int row) const;
    void set_value(int col, int row, int value);
    void reset_board();
    std::vector<int> get_row(int row) const;
    std::vector<int> get_column(int col) const;
    std::vector<std::vector<int> > get_all_downward_diagonals() const;
    std::vector<std::vector<int> > get_all_upward_diagonals() const;
    void update_captured_stone(const std::vector<std::pair<int, int> >& captured_stones);
    std::string convert_board_for_print() const;

private:
    int goal;
    int last_player;
    int next_player;
    int last_player_score;
    int next_player_score;
    std::vector<std::vector<int> > position;
};

#endif
