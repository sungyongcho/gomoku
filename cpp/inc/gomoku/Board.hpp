#ifndef BOARD_HPP
#define BOARD_HPP

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Gomoku.hpp"

struct CapturedStone {
  int x;
  int y;
  int player;
};

class Board {
 private:
  int goal;
  int last_player;
  int next_player;
  int last_player_score;
  int next_player_score;

  uint64_t last_player_board[BOARD_SIZE];
  uint64_t next_player_board[BOARD_SIZE];

  void resetBitboard();
  void initBitboardFromData(const std::vector<std::vector<char> > &board_data);

 public:
  std::vector<CapturedStone> captured_stones;
  Board();
  Board(const Board &other);
  Board(const std::vector<std::vector<char> > &board_data, int goal, const std::string &last_stone,
        const std::string &next_stone, int last_score, int next_score);

  std::pair<int, int> getCurrentScore();
  uint64_t *getBitboardByPlayer(int player);

  void setValueBit(int col, int row, int player);
  int getValueBit(int col, int row) const;

  void storeCapturedStone(int x, int y, int player);
  void applyCapture(bool clearCapture);

  // Fills the provided array with the occupancy (union of both players) for each row.
  void getOccupancy(uint64_t occupancy[BOARD_SIZE]) const;
  static Board *cloneBoard(const Board *board);

  static unsigned int getCellCount(unsigned int pattern, int windowLength);
  unsigned int extractLineAsBits(int x, int y, int dx, int dy, int length);

  int getLastPlayer();
  int getNextPlayer();

  int getLastPlayerScore();
  int getNextPlayerScore();

  void switchTurn();

  void printBitboard() const;
  void BitboardToJsonBoardboard(rapidjson::Value &json_board,
                                rapidjson::Document::AllocatorType &allocator) const;

  static std::string convertIndexToCoordinates(int col, int row);

  static void printLinePattern(unsigned int pattern, int length);
  static void printLinePatternReverse(unsigned int pattern, int length);
  static bool isValidCoordinate(int col, int row);
};

#endif  // BOARD_HPP
