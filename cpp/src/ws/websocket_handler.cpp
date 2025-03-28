#include "websocket_handler.hpp"

#include <cstring>
#include <ctime>  // Required for clock()

#include "Rules.hpp"
#include "minimax.hpp"

void responseSuccessMove(struct lws *wsi, Board &board, int aiPlayX, int aiPlayY,
                         double executionTime) {
  rapidjson::Document response;
  response.SetObject();
  rapidjson::Document::AllocatorType &allocator = response.GetAllocator();

  response.AddMember("type", "move", allocator);
  response.AddMember("status", "success", allocator);

  rapidjson::Value json_board(rapidjson::kArrayType);
  board.BitboardToJsonBoardboard(json_board, allocator);
  response.AddMember("board", json_board, allocator);

  rapidjson::Value lastPlay(rapidjson::kObjectType);
  {
    rapidjson::Value coordinate(rapidjson::kObjectType);
    coordinate.AddMember("x", aiPlayX, allocator);
    coordinate.AddMember("y", aiPlayY, allocator);
    lastPlay.AddMember("coordinate", coordinate, allocator);
    lastPlay.AddMember("stone", board.getNextPlayer() == 1 ? "X" : "O", allocator);
  }
  response.AddMember("lastPlay", lastPlay, allocator);

  // Build the capturedStones array.
  const std::vector<CapturedStone> &captured = board.getCapturedStones();
  rapidjson::Value capturedStones(rapidjson::kArrayType);
  for (size_t i = 0; i < captured.size(); ++i) {
    rapidjson::Value capturedObj(rapidjson::kObjectType);
    capturedObj.AddMember("x", captured[i].x, allocator);
    capturedObj.AddMember("y", captured[i].y, allocator);
    capturedObj.AddMember("stone", captured[i].player == 1 ? "X" : "O", allocator);
    capturedStones.PushBack(capturedObj, allocator);
  }
  response.AddMember("capturedStones", capturedStones, allocator);

  {
    double elapsed_ms = executionTime * 1000.0;
    double elapsed_ns = executionTime * 1e9;
    rapidjson::Value execTime(rapidjson::kObjectType);
    execTime.AddMember("s", executionTime, allocator);
    execTime.AddMember("ms", elapsed_ms, allocator);
    execTime.AddMember("ns", elapsed_ns, allocator);
    response.AddMember("executionTime", execTime, allocator);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  response.Accept(writer);
  std::string json_response = buffer.GetString();
  std::cout << "Json Response: " << json_response << std::endl;
  sendJsonResponse(wsi, json_response);
}

void responseSuccessEvaluate(struct lws *wsi, int evalScoreX, int evalScoreY) {
  rapidjson::Document response;
  response.SetObject();
  rapidjson::Document::AllocatorType &allocator = response.GetAllocator();

  response.AddMember("type", "evaluate", allocator);
  rapidjson::Value evalScores(rapidjson::kArrayType);

  // Assuming evalScoreY is for player "O" and evalScoreX for player "X"
  rapidjson::Value scoreO(rapidjson::kObjectType);
  scoreO.AddMember("player", "O", allocator);
  scoreO.AddMember("evalScore", evalScoreY, allocator);
  evalScores.PushBack(scoreO, allocator);

  rapidjson::Value scoreX(rapidjson::kObjectType);
  scoreX.AddMember("player", "X", allocator);
  scoreX.AddMember("evalScore", evalScoreX, allocator);
  evalScores.PushBack(scoreX, allocator);

  response.AddMember("evalScores", evalScores, allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  response.Accept(writer);
  std::string json_response = buffer.GetString();
  std::cout << "Json Response: " << json_response << std::endl;
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
        std::string difficulty;
        ParseResult result = parseMoveRequest(doc, pBoard, error, &last_x, &last_y, difficulty);

        if (result != PARSE_OK) {
          std::string error_response = constructErrorResponse(result, error);
          std::cout << error_response << std::endl;
          sendJsonResponse(wsi, error_response);
          return -1;
        }

        std::clock_t start = std::clock();  // Start time

        std::pair<int, int> a = Minimax::getBestMove(pBoard, difficulty == "easy" ? 1 : 5);

        std::clock_t end = std::clock();  // End time
        pBoard->setValueBit(a.first, a.second, pBoard->getNextPlayer());
        if (Rules::detectCaptureStones(*pBoard, a.first, a.second, pBoard->getNextPlayer())) {
          std::cout << "herher hrehreh" << std::endl;
          pBoard->applyCapture(false);
        }

        // // Calculate elapsed time
        double executionTime = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        // double elapsed_ms = executionTime * 1000.0;
        // double elapsed_ns = executionTime * 1e9;

        // std::cout << "Execution time: " << executionTime << " s, " << elapsed_ms << " ms, "
        //           << elapsed_ns << " ns" << std::endl;
        // // std::cout << a.first << ", " << a.second << std::endl;
        // std::cout << Board::convertIndexToCoordinates(a.first, a.second) << std::endl;
        // std::cout << "score: " << pBoard->getLastPlayerScore() << " , "
        //           << pBoard->getNextPlayerScore() << std::endl;

        // Minimax::simulateAIBattle(pBoard, 5, 80);

        responseSuccessMove(wsi, *pBoard, a.first, a.second, executionTime);
        delete pBoard;
        return 0;
      } else if (type == "evaluate") {
        Board *pBoard = NULL;
        std::string error;
        int eval_x;
        int eval_y;
        ParseResult result = parseEvaluateRequest(doc, pBoard, error, &eval_x, &eval_y);

        if (result != PARSE_OK) {
          std::string error_response = constructErrorResponse(result, error);
          std::cout << error_response << std::endl;
          sendJsonResponse(wsi, error_response);
          return -1;
        }
        // p1 mapped as x and p2 mapped as y
        int x_scores = Evaluation::evaluatePosition(pBoard, PLAYER_1, eval_x, eval_y);
        int y_scores = Evaluation::evaluatePosition(pBoard, PLAYER_2, eval_x, eval_y);

        std::cout << "x_scores: " << x_scores << " y_scores: " << y_scores << std::endl;
        responseSuccessEvaluate(wsi, x_scores, y_scores);

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
