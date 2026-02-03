#include "response_builder.hpp"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <sstream>

#include "Evaluation.hpp"

void responseSuccessMove(struct lws* wsi, Board& board, int aiPlayX, int aiPlayY,
                         double executionTime) {
  rapidjson::Document response;
  response.SetObject();
  rapidjson::Document::AllocatorType& allocator = response.GetAllocator();

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

  const std::vector<CapturedStone>& captured = board.getCapturedStones();
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
  sendJsonResponse(wsi, json_response);
}

void responseSuccessEvaluate(struct lws* wsi, int evalScoreX, int evalScoreY) {
  rapidjson::Document response;
  response.SetObject();
  rapidjson::Document::AllocatorType& allocator = response.GetAllocator();

  response.AddMember("type", "evaluate", allocator);
  rapidjson::Value evalScores(rapidjson::kArrayType);

  rapidjson::Value scoreO(rapidjson::kObjectType);
  scoreO.AddMember("player", "O", allocator);
  scoreO.AddMember("evalScore", evalScoreY, allocator);
  scoreO.AddMember("percentage", Evaluation::getEvaluationPercentage(evalScoreY), allocator);
  evalScores.PushBack(scoreO, allocator);

  rapidjson::Value scoreX(rapidjson::kObjectType);
  scoreX.AddMember("player", "X", allocator);
  scoreX.AddMember("evalScore", evalScoreX, allocator);
  scoreX.AddMember("percentage", Evaluation::getEvaluationPercentage(evalScoreX), allocator);
  evalScores.PushBack(scoreX, allocator);

  response.AddMember("evalScores", evalScores, allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  response.Accept(writer);
  std::string json_response = buffer.GetString();
  sendJsonResponse(wsi, json_response);
}

std::string constructErrorResponse(ParseResult result, const std::string& details) {
  std::ostringstream oss;
  oss << "{\"type\":\"error\",\"error\":\"";

  switch (result) {
    case ERROR_INVALID_BOARD:
      oss << "Invalid board field";
      break;
    case ERROR_INVALID_SCORES:
      oss << "Invalid scores field";
      break;
    case ERROR_GAME_DIFFICULTY:
      oss << "Game Difficulty Error";
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

void sendErrorResponse(struct lws* wsi, ParseResult result, const std::string& details) {
  std::string error_response = constructErrorResponse(result, details);
  sendJsonResponse(wsi, error_response);
}
