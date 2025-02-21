#include "websocket_handler.hpp"
#include "Board.hpp"
#include "Rules.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

Board *parse_json(struct lws *wsi, const rapidjson::Document &doc)
{
	if (!doc.HasMember("lastPlay"))
	{
		std::cerr << "AI first (no lastPlay found)" << std::endl;
		std::string response = "{\"type\":\"error\",\"status\":\"tba\"}";
		size_t resp_len = response.size();
		size_t buf_size = LWS_PRE + resp_len;
		unsigned char *buf = new unsigned char[buf_size];
		unsigned char *p = &buf[LWS_PRE];
		memcpy(p, response.c_str(), resp_len);
		lws_write(wsi, p, resp_len, LWS_WRITE_TEXT);
		delete[] buf;
		return NULL;
	}

	// Extract required fields
	int x = doc["lastPlay"]["coordinate"]["x"].GetInt();
	int y = doc["lastPlay"]["coordinate"]["y"].GetInt();
	std::string last_player = doc["lastPlay"]["stone"].GetString();
	std::string next_player = doc["nextPlayer"].GetString();
	int goal = doc["goal"].GetInt();

	std::cout << "Move received:" << std::endl;
	std::cout << "  Last Play: (" << x << ", " << y << ") by " << last_player << std::endl;
	std::cout << "  Next Player: " << next_player << std::endl;
	std::cout << "  Goal: " << goal << std::endl;

	// Convert board from JSON array to a 2D vector of char
	if (!doc.HasMember("board") || !doc["board"].IsArray())
	{
		std::cerr << "Error: Missing or invalid 'board' field." << std::endl;
		return NULL;
	}
	std::vector<std::vector<char> > board_data;
	for (rapidjson::SizeType i = 0; i < doc["board"].Size(); i++)
	{
		std::vector<char> row;
		for (rapidjson::SizeType j = 0; j < doc["board"][i].Size(); j++)
		{
			const char *cellStr = doc["board"][i][j].GetString();
			row.push_back(cellStr[0]);
		}
		board_data.push_back(row);
	}

	// Extract scores
	if (!doc.HasMember("scores") || !doc["scores"].IsArray())
	{
		std::cerr << "Error: Missing or invalid 'scores' field." << std::endl;
		return NULL;
	}
	int last_player_score = 0;
	int next_player_score = 0;
	for (rapidjson::SizeType i = 0; i < doc["scores"].Size(); i++)
	{
		std::string player = doc["scores"][i]["player"].GetString();
		int score = doc["scores"][i]["score"].GetInt();
		if (player == "X")
		{
			last_player_score = score;
		}
		else if (player == "O")
		{
			next_player_score = score;
		}
	}

	// Create a new Board instance
	Board *pBoard = new Board(board_data, goal, last_player, next_player,
							  last_player_score, next_player_score);

	std::cout << "Parsed Board State:\n"
			  << pBoard->convert_board_for_print() << std::endl;

	// (Optional) Debug prints of nearby board values
	std::cout << pBoard->get_value(x, y) << std::endl;
	std::cout << pBoard->get_value(x - 1, y) << std::endl;
	std::cout << pBoard->get_value(x - 2, y) << std::endl;
	std::cout << pBoard->get_value(x - 3, y) << std::endl;

	// Process captures
	std::vector<std::pair<int, int> > captured =
		Rules::capture_opponent(*pBoard, x, y, (last_player == "X") ? PLAYER_1 : PLAYER_2);
	if (!captured.empty())
	{
		std::cout << "Captured stones:" << std::endl;
		for (std::vector<std::pair<int, int> >::iterator it = captured.begin();
			 it != captured.end(); ++it)
		{
			std::cout << " - (" << it->first << ", " << it->second << ")" << std::endl;
		}
		std::cout << std::flush;
		Rules::remove_captured_stone(*pBoard, captured);
	}
	else
	{
		std::cout << "No captures made." << std::endl;
	}

	if (Rules::double_three_detected(*pBoard, x, y, (last_player == "X") ? PLAYER_1 : PLAYER_2))
	{
		std::cout << "doublethree detected" << std::endl;
		// await websocket.send_json({"type": "error", "error": "doublethree"})
		std::string response = "{\"type\":\"error\",\"error\":\"doublethree\"}";
		size_t resp_len = response.size();
		size_t buf_size = LWS_PRE + resp_len;
		unsigned char *buf = new unsigned char[buf_size];
		unsigned char *p = &buf[LWS_PRE];
		memcpy(p, response.c_str(), resp_len);
		lws_write(wsi, p, resp_len, LWS_WRITE_TEXT);
		delete[] buf;
		return NULL;

	}
	return pBoard;
}

void success_response(struct lws *wsi, Board &board)
{
	rapidjson::Document response;
	response.SetObject();
	rapidjson::Document::AllocatorType &allocator = response.GetAllocator();

	response.AddMember("type", "move", allocator);
	response.AddMember("status", "success", allocator);

	rapidjson::Value json_board(rapidjson::kArrayType);
	board.to_json_board(json_board, allocator);
	response.AddMember("board", json_board, allocator);

	response.AddMember("scores", "success", allocator);
	response.AddMember("lastPlay", "success", allocator);
	response.AddMember("capturedStones", "success", allocator);

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	response.Accept(writer);
	std::string json_response = buffer.GetString();

	std::cout << json_response << std::endl;

	// Dynamically allocate a buffer sized to LWS_PRE plus the JSON response length
	size_t json_length = json_response.size();
	size_t buf_size = LWS_PRE + json_length;
	unsigned char *buf = new unsigned char[buf_size];
	unsigned char *p = &buf[LWS_PRE];

	memcpy(p, json_response.c_str(), json_length);
	lws_write(wsi, p, json_length, LWS_WRITE_TEXT);
	delete[] buf;
}

int callback_debug(struct lws *wsi, enum lws_callback_reasons reason,
				   void *user, void *in, size_t len)
{
	(void)user;
	switch (reason)
	{
	case LWS_CALLBACK_ESTABLISHED:
		std::cout << "WebSocket `/ws/debug` connected!" << std::endl;
		break;

	case LWS_CALLBACK_RECEIVE:
	{
		std::string received_msg((char *)in, len);
		std::cout << "Received: " << received_msg << std::endl;

		rapidjson::Document doc;
		if (doc.Parse(received_msg.c_str()).HasParseError())
		{
			std::cerr << "JSON Parse Error!" << std::endl;
			return -1;
		}
		if (!doc.HasMember("type") || !doc["type"].IsString())
		{
			std::cerr << "Error: Missing or invalid 'type' field." << std::endl;
			return -1;
		}
		std::string type = doc["type"].GetString();
		if (type == "move")
		{
			Board *pBoard = parse_json(wsi, doc);
			if (!pBoard)
			{
				return -1;
			}
			success_response(wsi, *pBoard);
			delete pBoard;
			return 0;
		}
		else
		{
			std::cerr << "Unknown type: " << type << std::endl;
			return -1;
		}
		break;
	}

	case LWS_CALLBACK_CLOSED:
		std::cout << "WebSocket connection closed." << std::endl;
		break;

	default:
		break;
	}
	return 0;
}
