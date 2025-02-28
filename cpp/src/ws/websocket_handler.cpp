#include "websocket_handler.hpp"
#include "Board.hpp"
#include "Rules.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

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
	send_json_response(wsi, json_response);
}

std::string construct_error_response(ParseResult result, const std::string &details)
{
	std::ostringstream oss;
	oss << "{\"type\":\"error\",\"error\":\"";

	switch (result)
	{
	case ERROR_NO_LAST_PLAY:
		oss << "No lastPlay found";
		break;
	case ERROR_INVALID_BOARD:
		oss << "Invalid board field";
		break;
	case ERROR_INVALID_SCORES:
		oss << "Invalid scores field";
		break;
	case ERROR_DOUBLE_THREE:
		oss << "doublethree";
		break;
	default:
		oss << "Unknown error";
		break;
	}

	if (!details.empty())
	{
		oss << ": " << details;
	}

	oss << "\"}";
	return oss.str();
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
			std::string error_response = construct_error_response(ERROR_UNKNOWN, "JSON Parse Error");
			send_json_response(wsi, error_response);
			return -1;
		}

		if (!doc.HasMember("type") || !doc["type"].IsString())
		{
			std::string error_response = construct_error_response(ERROR_UNKNOWN, "Invalid 'type' field");
			send_json_response(wsi, error_response);
			return -1;
		}

		std::string type = doc["type"].GetString();
		if (type == "move")
		{
			Board *pBoard = NULL;
			std::string error;
			ParseResult result = parse_json(doc, pBoard, error);

			if (result != PARSE_OK)
			{
				std::string error_response = construct_error_response(result, error);
				std::cout << error_response << std::endl;
				std::cout << "-------------"<< std::endl;
				send_json_response(wsi, error_response);
				return -1;
			}

			success_response(wsi, *pBoard);
			delete pBoard;
			return 0;
		}
		else
		{
			std::string error_response = construct_error_response(ERROR_UNKNOWN, "Unknown type");
			send_json_response(wsi, error_response);
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
