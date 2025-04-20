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

  std::vector<CapturedStone> captured_stones;

  uint64_t currentHash;

  void resetBitboard();
  void initBitboardFromData(const std::vector<std::vector<char> > &board_data);
  void updateLastPlayerScore(int newScore);
  void updateNextPlayerScore(int newScore);

 public:
  Board();
  Board(const Board &other);
  Board(const std::vector<std::vector<char> > &board_data, int goal, int last_player_int,
        int next_player_int, int last_score, int next_score);

  std::pair<int, int> getCurrentScore();
  uint64_t *getBitboardByPlayer(int player);

  void setValueBit(int col, int row, int stone);
  int getValueBit(int col, int row) const;

  void storeCapturedStone(int x, int y, int player);
  void applyCapture(bool clearCapture);

  // Fills the provided array with the occupancy (union of both players) for each row.
  void getOccupancy(uint64_t occupancy[BOARD_SIZE]) const;

  static unsigned int getCellCount(unsigned int pattern, int windowLength);
  unsigned int extractLineAsBits(int x, int y, int dx, int dy, int length);

  int getLastPlayer();
  int getNextPlayer();

  int getLastX();
  int getLastY();
  int getLastEvalScore();

  void setLastEvalScore(int score);
  int getLastPlayerScore();
  int getNextPlayerScore();

  int getGoal();

  uint64_t getHash();

  const std::vector<CapturedStone> &getCapturedStones() const;

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
