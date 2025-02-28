#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <rapidjson/document.h>
#include "Board.hpp"
#include "Rules.hpp"
#include <vector>
#include <string>
#include <utility>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <libwebsockets.h>

enum ParseResult
{
	PARSE_OK,
	ERROR_NO_LAST_PLAY,
	ERROR_INVALID_BOARD,
	ERROR_INVALID_SCORES,
	ERROR_DOUBLE_THREE,
	ERROR_UNKNOWN
};

bool extract_required_fields(const rapidjson::Document &doc, int &x, int &y,
							 std::string &last_player, std::string &next_player, int &goal);
bool parse_board(const rapidjson::Document &doc, std::vector<std::vector<char> > &board_data);
bool parse_scores(const rapidjson::Document &doc, int &last_player_score, int &next_player_score);
std::vector<std::vector<char> > parse_board_from_json(const rapidjson::Document &doc);
void handle_captures(Board &board, int x, int y, const std::string &last_player);

void send_json_response(struct lws *wsi, const std::string &response);
ParseResult parse_json(const rapidjson::Document &doc, Board *&pBoard, std::string &error);

#endif // JSON_PARSER_H
