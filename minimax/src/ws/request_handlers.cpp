#include "request_handlers.hpp"

#include <ctime>
#include <iostream>

#include "Evaluation.hpp"
#include "Minimax.hpp"
#include "Rules.hpp"
#include "json_parser.hpp"
#include "response_builder.hpp"

namespace {

std::pair<int, int> selectBestMove(Board* board, int last_x, int last_y,
                                   const std::string& difficulty) {
  if (last_x == -1 && last_y == -1) {
    std::cout << board->getLastPlayer() << " " << board->getNextPlayer() << std::endl;
    std::cout << "no lastplay" << std::endl;
    return std::make_pair(BOARD_SIZE / 2, BOARD_SIZE / 2);
  }

  if (difficulty == "hard")
    return Minimax::getBestMovePVS(board, MAX_DEPTH, &Evaluation::evaluatePositionHard);
  if (difficulty == "medium")
    return Minimax::iterativeDeepening(board, MAX_DEPTH, 0.4, &Evaluation::evaluatePosition);
  if (difficulty == "easy") return Minimax::getBestMove(board, 5, &Evaluation::evaluatePosition);

  return std::make_pair(-1, -1);
}

void applyMoveAndCapture(Board* board, int x, int y) {
  board->setValueBit(x, y, board->getNextPlayer());
  if (Rules::detectCaptureStones(*board, x, y, board->getNextPlayer())) {
    board->applyCapture(false);
  }
}

double computeExecutionTimeSeconds(std::clock_t start, std::clock_t end) {
  return static_cast<double>(end - start) / CLOCKS_PER_SEC;
}

}  // namespace

int handleMoveRequest(struct lws* wsi, const rapidjson::Document& doc, psd_debug* psd) {
  Board* pBoard = NULL;
  std::string error;
  int last_x;
  int last_y;
  std::string difficulty;
  std::pair<int, int> predict;

  ParseResult result = parseMoveRequest(doc, pBoard, error, &last_x, &last_y, difficulty);

  if (result != PARSE_OK) {
    std::cout << constructErrorResponse(result, error) << std::endl;
    sendErrorResponse(wsi, result, error);
    return -1;
  }

  if (psd->difficulty != difficulty) {
    if (psd->difficulty.size() == 0)
      std::cout << "initial difficulty: " << difficulty << std::endl;
    else {
      std::cout << "difficulty changed from" << psd->difficulty << " to " << difficulty
                << std::endl;

      transTable.clear();
    }
    psd->difficulty = difficulty;
  }

  std::clock_t start = std::clock();
  predict = selectBestMove(pBoard, last_x, last_y, difficulty);
  if (predict.first == -1 && predict.second == -1) {
    std::string error_response = constructErrorResponse(ERROR_GAME_DIFFICULTY, "");
    std::cout << error_response << std::endl;
    sendJsonResponse(wsi, error_response);
    return -1;
  }

  std::clock_t end = std::clock();
  applyMoveAndCapture(pBoard, predict.first, predict.second);

  double executionTime = computeExecutionTimeSeconds(start, end);
  double elapsed_ms = executionTime * 1000.0;
  double elapsed_ns = executionTime * 1e9;

  std::cout << "Execution time: " << executionTime << " s, " << elapsed_ms << " ms, " << elapsed_ns
            << " ns" << std::endl;

  responseSuccessMove(wsi, *pBoard, predict.first, predict.second, executionTime);
  delete pBoard;
  return 0;
}

int handleEvaluateRequest(struct lws* wsi, const rapidjson::Document& doc) {
  Board* pBoard = NULL;
  std::string error;
  int eval_x;
  int eval_y;
  ParseResult result = parseEvaluateRequest(doc, pBoard, error, &eval_x, &eval_y);

  if (result != PARSE_OK) {
    std::cout << constructErrorResponse(result, error) << std::endl;
    sendErrorResponse(wsi, result, error);
    return -1;
  }

  int x_scores = Evaluation::evaluatePositionHard(pBoard, PLAYER_1, eval_x, eval_y);
  int o_scores = Evaluation::evaluatePositionHard(pBoard, PLAYER_2, eval_x, eval_y);

  int x_percentage = Evaluation::getEvaluationPercentage(x_scores);
  int o_percentage = Evaluation::getEvaluationPercentage(o_scores);
  std::cout << "x_scores: " << x_scores << " y_scores: " << o_scores << std::endl;
  std::cout << "x_percentage: " << x_percentage << " y_percentage: " << o_percentage << std::endl;
  responseSuccessEvaluate(wsi, x_scores, o_scores);

  delete pBoard;
  return 0;
}

int handleTestRequest(struct lws* wsi, const rapidjson::Document& doc) {
  initZobrist();
  transTable.clear();

  Board* pBoard = NULL;
  std::string error;
  int last_x;
  int last_y;
  std::string difficulty;
  ParseResult result = parseMoveRequest(doc, pBoard, error, &last_x, &last_y, difficulty);

  if (result != PARSE_OK) {
    std::cout << constructErrorResponse(result, error) << std::endl;
    sendErrorResponse(wsi, result, error);
    return -1;
  }

  std::clock_t start = std::clock();
  std::pair<int, int> a =
      Minimax::getBestMovePVS(pBoard, MAX_DEPTH, &Evaluation::evaluatePositionHard);
  std::clock_t end = std::clock();

  applyMoveAndCapture(pBoard, a.first, a.second);

  double executionTime = computeExecutionTimeSeconds(start, end);
  double elapsed_ms = executionTime * 1000.0;
  double elapsed_ns = executionTime * 1e9;

  std::cout << "Execution time: " << executionTime << " s, " << elapsed_ms << " ms, " << elapsed_ns
            << " ns" << std::endl;

  responseSuccessMove(wsi, *pBoard, a.first, a.second, executionTime);
  delete pBoard;
  return 0;
}

void handleResetRequest(psd_debug* psd) {
  psd->difficulty = "";
  initZobrist();
  transTable.clear();
}
