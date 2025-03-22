#include "websocket_handler.hpp"

#include <cstring>
#include <ctime>  // Required for clock()
#include <iostream>
#include <string>
#include <vector>

#include "Board.hpp"
#include "Rules.hpp"
#include "minimax.hpp"

void responseSuccess(struct lws *wsi, Board &board) {
  rapidjson::Document response;
  response.SetObject();
  rapidjson::Document::AllocatorType &allocator = response.GetAllocator();

  response.AddMember("type", "move", allocator);
  response.AddMember("status", "success", allocator);

  rapidjson::Value json_board(rapidjson::kArrayType);
  board.BitboardToJsonBoardboard(json_board, allocator);
  response.AddMember("board", json_board, allocator);

  response.AddMember("scores", "success", allocator);
  response.AddMember("lastPlay", "success", allocator);
  response.AddMember("capturedStones", "success", allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  response.Accept(writer);
  std::string json_response = buffer.GetString();
  sendJsonResponse(wsi, json_response);
}

std::string constructErrorResponse(ParseResult result, const std::string &details) {
  std::ostringstream oss;
  oss << "{\"type\":\"error\",\"error\":\"";

  switch (result) {
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

  if (!details.empty()) {
    oss << ": " << details;
  }

  oss << "\"}";
  return oss.str();
}

int callbackDebug(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in,
                  size_t len) {
  (void)user;
  switch (reason) {
    case LWS_CALLBACK_ESTABLISHED:
      std::cout << "WebSocket `/ws/debug` connected!" << std::endl;
      break;

    case LWS_CALLBACK_RECEIVE: {
      std::string received_msg((char *)in, len);
      std::cout << "Received: " << received_msg << std::endl;

      rapidjson::Document doc;
      if (doc.Parse(received_msg.c_str()).HasParseError()) {
        std::string error_response = constructErrorResponse(ERROR_UNKNOWN, "JSON Parse Error");
        sendJsonResponse(wsi, error_response);
        return -1;
      }

      if (!doc.HasMember("type") || !doc["type"].IsString()) {
        std::string error_response = constructErrorResponse(ERROR_UNKNOWN, "Invalid 'type' field");
        sendJsonResponse(wsi, error_response);
        return -1;
      }

      std::string type = doc["type"].GetString();
      if (type == "move") {
        Board *pBoard = NULL;
        std::string error;
        int last_x;
        int last_y;
        ParseResult result = parseJson(doc, pBoard, error, &last_x, &last_y);

        if (result != PARSE_OK) {
          // TODO: revert back to doublethree and send back
          std::string error_response = constructErrorResponse(result, error);
          std::cout << error_response << std::endl;
          sendJsonResponse(wsi, error_response);
          if (result == ERROR_DOUBLE_THREE) return 0;
          return -1;
        }

        std::cout << "coordinates: " << Board::convertIndexToCoordinates(last_x, last_y)
                  << std::endl;
        // int check = Minimax::evaluatePosition(pBoard, pBoard->getLastPlayer(), last_x, last_y);

        // std::cout << "score: " << check << std::endl;

        // std::vector<std::pair<int, int> > candidates1 = Minimax::generateCandidateMoves(pBoard);

        // std::cout << "\nCandidate Moves (Row-based):" << std::endl;
        // for (size_t i = 0; i < candidates1.size(); i++) {
        //   std::cout << "(" << candidates1[i].first << ", " << candidates1[i].second << ")"
        //             << std::endl;
        // }

        // Minimax::printBoardWithCandidates(pBoard, candidates1);
        std::clock_t start = std::clock();  // Start time

        std::pair<int, int> a = Minimax::getBestMove(pBoard, 5);

        std::clock_t end = std::clock();  // End time

        // Calculate elapsed time
        double elapsed_seconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        double elapsed_ms = elapsed_seconds * 1000.0;
        double elapsed_ns = elapsed_seconds * 1e9;

        std::cout << "Execution time: " << elapsed_seconds << " s, " << elapsed_ms << " ms, "
                  << elapsed_ns << " ns" << std::endl;

        std::cout << a.first << ", " << a.second << std::endl;
        std::cout << Board::convertIndexToCoordinates(a.first, a.second) << std::endl;

        // Minimax::simulateAIBattle(pBoard, 5, 80);

        responseSuccess(wsi, *pBoard);
        delete pBoard;
        return 0;
      } else {
        std::string error_response = constructErrorResponse(ERROR_UNKNOWN, "Unknown type");
        sendJsonResponse(wsi, error_response);
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
