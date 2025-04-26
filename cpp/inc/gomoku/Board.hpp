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

  void reset_bitboard();
  void init_bitboard_from_data(const std::vector<std::vector<char> > &board_data);

 public:
  // Constructors
  Board();
  Board(const Board &other);
  Board(const std::vector<std::vector<char> > &board_data, int goal, int last_player_int,
        int next_player_int, int last_score, int next_score);
  Board(int goal, int last_player_int, int next_player_int, int last_score, int next_score);

  // Board State & Accessors
  int getValueBit(int col, int row) const;                  // Get stone type at (col, row)
  uint64_t *getBitboardByPlayer(int player);                // Get pointer to player's bitboard
  void getOccupancy(uint64_t occupancy[BOARD_SIZE]) const;  // Get combined occupancy
  uint64_t getHash() const;                                 // Get the current Zobrist hash

  // Player & Score Information
  int getLastPlayer() const;  // Player who made the last move
  int getNextPlayer() const;  // Player whose turn it is now
  int getLastPlayerScore() const;
  int getNextPlayerScore() const;
  std::pair<int, int> getCurrentScore()
      const;            // Returns {next_player_score, last_player_score} ? Clarify order if needed.
  int getGoal() const;  // Score needed to win

  // Move Execution & Captures
  void setValueBit(int col, int row, int stone);      // Place a stone (updates hash internally)
  void storeCapturedStone(int x, int y, int player);  // Record a capture event
  void applyCapture(bool clearCaptureList);           // Process stored captures (updates hash)
  const std::vector<CapturedStone> &getCapturedStones()
      const;          // Get list of captures from last move
  void switchTurn();  // Switches players (updates hash)

  // Utility & Static Methods
  static bool isValidCoordinate(int col, int row);
  static std::string convertIndexToCoordinates(int col, int row);
  unsigned int extractLineAsBits(int x, int y, int dx, int dy, int length) const;
  static unsigned int getCellCount(unsigned int pattern, int windowLength);

  // For debug
  void printBitboard() const;
  void BitboardToJsonBoardboard(rapidjson::Value &json_board,
                                rapidjson::Document::AllocatorType &allocator) const;
  static void printLinePattern(unsigned int pattern, int length);
  static void printLinePatternReverse(unsigned int pattern, int length);
};

#endif  // BOARD_HPP
