#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <libwebsockets.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <sstream>  // Required for std::stringstream
#include <string>
#include <utility>
#include <vector>

#include "Board.hpp"
#include "Rules.hpp"

enum ParseResult {
  PARSE_OK,
  ERROR_INVALID_BOARD,
  ERROR_INVALID_SCORES,
  ERROR_GAME_DIFFICULTY,
  ERROR_UNKNOWN
};

bool extractMoveFields(const rapidjson::Document &doc, int &x, int &y, std::string &last_player,
                       std::string &next_player, int &goal, bool &enable_capture,
                       bool &enable_double_three_restriction);

bool extractEvaluationFields(const rapidjson::Document &doc, int &x, int &y,
                             std::string &last_player, std::string &next_player, int &goal,
                             std::string &difficulty, bool &enable_capture,
                             bool &enable_double_three_restriction);

bool parseBoard(const rapidjson::Document &doc, std::vector<std::vector<char> > &board_data);
bool parseScores(const rapidjson::Document &doc, std::string last_player, std::string next_player,
                 int &last_player_score, int &next_player_score);
std::vector<std::vector<char> > parseBoardFromJson(const rapidjson::Document &doc);

void sendJsonResponse(struct lws *wsi, const std::string &response);
ParseResult parseMoveRequest(const rapidjson::Document &doc, Board *&pBoard, std::string &error,
                             int *last_x, int *last_y, std::string &difficulty);

ParseResult parseEvaluateRequest(const rapidjson::Document &doc, Board *&pBoard, std::string &error,
                                 int *last_x, int *last_y);

#endif  // JSON_PARSER_H
