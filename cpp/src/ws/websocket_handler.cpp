#include "websocket_handler.hpp"

#include <cstring>
#include <ctime>  // Required for clock()

#include "Minimax.hpp"
#include "Rules.hpp"

// void printTestState(const char *description, const Board &board) {
//   std::cout << description << std::endl;
//   // board.printBitboard(); // Optional: print board state visually
//   std::cout << "  Hash: " << board.getHash() << ", P1 Score: "
//             << (board.getLastPlayer() == PLAYER_1 ? board.getLastPlayerScore()
//                                                   : board.getNextPlayerScore())  // Get P1 score
//             << ", P2 Score: "
//             << (board.getLastPlayer() == PLAYER_2 ? board.getLastPlayerScore()
//                                                   : board.getNextPlayerScore())  // Get P2 score
//             << ", Next Player: " << board.getNextPlayer() << std::endl;
// }

// // Main test function
// void testZobristHashingLogic() {
//   std::cout << "\n=======================================" << std::endl;
//   std::cout << "  Testing Zobrist Hashing Logic" << std::endl;
//   std::cout << "=======================================" << std::endl;

//   // Ensure keys are initialized (call initZobrist() once before this test)
//   assert(Zobrist::initialized && "Zobrist keys must be initialized before testing!");

//   // --- Test Case 1: Initial State & Piece Placement/Removal ---
//   std::cout << "\n--- Test Case 1: Initialization & Piece XOR ---" << std::endl;
//   Board board1;  // Uses default constructor
//   ZobristKey expected_hash = 0;
//   expected_hash ^= Zobrist::capture_keys[PLAYER_1][0];  // P1 score 0
//   expected_hash ^= Zobrist::capture_keys[PLAYER_2][0];  // P2 score 0
//   if (board1.getNextPlayer() == PLAYER_2) {             // Check if P2 starts by default
//     expected_hash ^= Zobrist::turn_key;
//   }
//   printTestState("1a: Initial Board", board1);
//   assert(board1.getHash() == expected_hash && "Initial hash incorrect!");

//   // Place P1 at (9, 9)
//   board1.setValueBit(9, 9, PLAYER_1);
//   expected_hash ^= Zobrist::piece_keys[9][9][PLAYER_1];
//   printTestState("1b: Place P1(9,9)", board1);
//   assert(board1.getHash() == expected_hash && "Hash after P1 place incorrect!");

//   // Place P2 at (0, 0)
//   board1.setValueBit(0, 0, PLAYER_2);
//   expected_hash ^= Zobrist::piece_keys[0][0][PLAYER_2];
//   printTestState("1c: Place P2(0,0)", board1);
//   assert(board1.getHash() == expected_hash && "Hash after P2 place incorrect!");

//   // Remove P1 from (9, 9)
//   board1.setValueBit(9, 9, EMPTY_SPACE);
//   expected_hash ^= Zobrist::piece_keys[9][9][PLAYER_1];  // XOR out P1 key
//   printTestState("1d: Remove P1(9,9)", board1);
//   assert(board1.getHash() == expected_hash && "Hash after P1 remove incorrect!");

//   // Remove P2 from (0, 0)
//   board1.setValueBit(0, 0, EMPTY_SPACE);
//   expected_hash ^= Zobrist::piece_keys[0][0][PLAYER_2];  // XOR out P2 key
//   printTestState("1e: Remove P2(0,0)", board1);
//   assert(board1.getHash() == expected_hash && "Hash after P2 remove incorrect!");

//   // Check if hash returned to initial empty state (with correct turn)
//   ZobristKey initial_empty_hash = 0;
//   initial_empty_hash ^= Zobrist::capture_keys[PLAYER_1][0];
//   initial_empty_hash ^= Zobrist::capture_keys[PLAYER_2][0];
//   if (board1.getNextPlayer() == PLAYER_2) initial_empty_hash ^= Zobrist::turn_key;
//   assert(board1.getHash() == initial_empty_hash && "Hash did not revert to initial empty
//   state!"); std::cout << "Piece Placement/Removal Test: PASSED" << std::endl;

//   // --- Test Case 2: Turn Switching ---
//   std::cout << "\n--- Test Case 2: Turn Switch XOR ---" << std::endl;
//   Board board2;  // Fresh board
//   ZobristKey hash_before_switch = board2.getHash();
//   printTestState("2a: Before Switch", board2);

//   board2.switchTurn();
//   ZobristKey hash_after_switch1 = board2.getHash();
//   printTestState("2b: After 1st Switch", board2);
//   assert(hash_after_switch1 == (hash_before_switch ^ Zobrist::turn_key) &&
//          "Hash incorrect after 1st switch!");

//   board2.switchTurn();  // Switch back
//   ZobristKey hash_after_switch2 = board2.getHash();
//   printTestState("2c: After 2nd Switch", board2);
//   assert(hash_after_switch2 == hash_before_switch && "Hash did not revert after 2nd switch!");
//   std::cout << "Turn Switch Test: PASSED" << std::endl;

//   // --- Test Case 3: Capture (Score Change + Piece Removal) ---
//   std::cout << "\n--- Test Case 3: Capture XOR ---" << std::endl;
//   Board board3(5, PLAYER_1, PLAYER_2, 0, 0);  // Fresh board
//   // Setup: P1 captures two P2 stones. Assume P1 just moved.
//   board3.setValueBit(4, 4, PLAYER_2);  // Place P2 stone (hash updated for piece)
//   board3.setValueBit(5, 5, PLAYER_2);  // Place P2 stone (hash updated for piece)

//   // Manually recalculate expected hash *before* applying capture
//   expected_hash = 0;
//   expected_hash ^= Zobrist::piece_keys[4][4][PLAYER_2];
//   expected_hash ^= Zobrist::piece_keys[5][5][PLAYER_2];
//   expected_hash ^= Zobrist::capture_keys[PLAYER_1][0];
//   expected_hash ^= Zobrist::capture_keys[PLAYER_2][0];
//   if (board3.getNextPlayer() == PLAYER_2)
//     expected_hash ^= Zobrist::turn_key;  // Add turn key if needed
//   // If the board's hash doesn't match this after setup, setValueBit has issues
//   // assert(board3.getHash() == expected_hash && "Pre-capture hash setup failed!");
//   // Re-sync expected hash with board's reality before testing applyCapture
//   expected_hash = board3.getHash();
//   printTestState("3a: Before Capture Applied", board3);

//   // Simulate the capture occurring
//   board3.storeCapturedStone(4, 4, PLAYER_2);  // P2's stone
//   board3.storeCapturedStone(5, 5, PLAYER_2);  // P2's stone

//   // Apply the capture: removes pieces, updates score, updates hash
//   board3.applyCapture(true);

//   // Calculate expected hash changes caused *by applyCapture*:
//   // 1. Piece removal for P2 at (4,4)
//   expected_hash ^= Zobrist::piece_keys[4][4][PLAYER_2];
//   // 2. Piece removal for P2 at (5,5)
//   expected_hash ^= Zobrist::piece_keys[5][5][PLAYER_2];
//   // 3. Score change for P1 (capturing player: last_player) from 0 to 1
//   expected_hash ^= Zobrist::capture_keys[PLAYER_1][0];  // XOR out score 0 key
//   expected_hash ^= Zobrist::capture_keys[PLAYER_1][1];  // XOR in score 1 key

//   printTestState("3b: After Capture Applied", board3);
//   assert(board3.getHash() == expected_hash && "Hash incorrect after capture!");
//   assert(board3.getLastPlayerScore() == 1 && "P1 score incorrect after capture!");
//   assert(board3.getNextPlayerScore() == 0 && "P2 score changed incorrectly!");
//   std::cout << "Capture Test: PASSED" << std::endl;

//   std::cout << "\n--- Zobrist Hash Logic Test Finished ---" << std::endl;
// }

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
  // std::cout << "Json Response: " << json_response << std::endl;
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
  // std::cout << "Json Response: " << json_response << std::endl;
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
      initZobrist();
      // testZobristHashingLogic();
      transTable.clear();
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

        // std::clock_t start = std::clock();  // Start time

        // std::clock_t end = std::clock();  // End time
        // pBoard->setValueBit(a.first, a.second, pBoard->getNextPlayer());
        // if (Rules::detectCaptureStones(*pBoard, a.first, a.second, pBoard->getNextPlayer())) {
        //   std::cout << "herher hrehreh" << std::endl;
        //   pBoard->applyCapture(false);
        // }

        // // Calculate elapsed time
        // double executionTime = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        std::clock_t start = std::clock();  // Start time

        // std::pair<int, int> a = Minimax::getBestMove(pBoard, difficulty == "easy" ? 1 :
        // MAX_DEPTH);
        std::pair<int, int> a = Minimax::iterativeDeepening(pBoard, 8, .5);
        //     Minimax::iterativeDeepening(pBoard, 1, 500);

        std::clock_t end = std::clock();  // End time
        pBoard->setValueBit(a.first, a.second, pBoard->getNextPlayer());
        if (Rules::detectCaptureStones(*pBoard, a.first, a.second, pBoard->getNextPlayer())) {
          pBoard->applyCapture(false);
        }

        // // Calculate elapsed time
        double executionTime = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        double elapsed_ms = executionTime * 1000.0;
        double elapsed_ns = executionTime * 1e9;

        std::cout << "Execution time: " << executionTime << " s, " << elapsed_ms << " ms, "
                  << elapsed_ns << " ns" << std::endl;

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

        // pBoard->setLastEvalScore(Evaluation::evaluatePositionHard(
        //     pBoard, pBoard->getLastPlayer(), pBoard->getLastX(), pBoard->getLastY()));

        // p1 mapped as x and p2 mapped as o
        int x_scores = Evaluation::evaluatePositionHard(pBoard, PLAYER_1, eval_x, eval_y);
        int o_scores = Evaluation::evaluatePositionHard(pBoard, PLAYER_2, eval_x, eval_y);

        int x_percentage = Evaluation::getEvaluationPercentage(x_scores);
        int o_percentage = Evaluation::getEvaluationPercentage(o_scores);
        std::cout << "x_scores: " << x_scores << " y_scores: " << o_scores << std::endl;
        std::cout << "x_percentage: " << x_percentage << " y_percentage: " << o_percentage
                  << std::endl;
        responseSuccessEvaluate(wsi, x_scores, o_scores);

        delete pBoard;
        return 0;
      } else if (type == "reset") {
        initZobrist();
        transTable.clear();
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
