#include "json_parser.hpp"

void sendJsonResponse(struct lws *wsi, const std::string &response) {
  size_t resp_len = response.size();
  size_t buf_size = LWS_PRE + resp_len;
  unsigned char *buf = new unsigned char[buf_size];
  unsigned char *p = &buf[LWS_PRE];
  memcpy(p, response.c_str(), resp_len);
  lws_write(wsi, p, resp_len, LWS_WRITE_TEXT);
  delete[] buf;
}

bool extractRequiredFields(const rapidjson::Document &doc, int &x, int &y, std::string &last_player,
                           std::string &next_player, int &goal) {
  x = doc["lastPlay"]["coordinate"]["x"].GetInt();
  y = doc["lastPlay"]["coordinate"]["y"].GetInt();
  last_player = doc["lastPlay"]["stone"].GetString();
  next_player = doc["nextPlayer"].GetString();
  goal = doc["goal"].GetInt();

  std::cout << "Move received:" << std::endl;
  std::cout << "  Last Play: (" << x << ", " << y << ") by " << last_player << std::endl;
  std::cout << "  Next Player: " << next_player << std::endl;
  std::cout << "  Goal: " << goal << std::endl;

  return true;
}

std::vector<std::vector<char> > parseBoardFromJson(const rapidjson::Document &doc) {
  std::vector<std::vector<char> > board_data;

  for (rapidjson::SizeType i = 0; i < doc["board"].Size(); i++) {
    std::vector<char> row;
    for (rapidjson::SizeType j = 0; j < doc["board"][i].Size(); j++) {
      const char *cellStr = doc["board"][i][j].GetString();
      row.push_back(cellStr[0]);
    }
    board_data.push_back(row);
  }
  return board_data;
}

bool parseBoard(const rapidjson::Document &doc, std::vector<std::vector<char> > &board_data) {
  if (!doc.HasMember("board") || !doc["board"].IsArray()) {
    std::cerr << "Error: Missing or invalid 'board' field." << std::endl;
    return false;
  }

  board_data = parseBoardFromJson(doc);
  return true;
}

bool parseScores(const rapidjson::Document &doc, std::string last_player, std::string next_player,
                 int &last_player_score, int &next_player_score) {
  if (!doc.HasMember("scores") || !doc["scores"].IsArray()) {
    std::cerr << "Error: Missing or invalid 'scores' field." << std::endl;
    return false;
  }

  for (rapidjson::SizeType i = 0; i < doc["scores"].Size(); i++) {
    std::string player = doc["scores"][i]["player"].GetString();
    int score = doc["scores"][i]["score"].GetInt();
    if (player == last_player)
      last_player_score = score;
    else if (player == next_player)
      next_player_score = score;
  }

  return true;
}

// Main parseJson function using ParseResult enum
ParseResult parseJson(const rapidjson::Document &doc, Board *&pBoard, std::string &error,
                      // temporary for storing the position
                      int *last_x, int *last_y) {
  int x, y, goal;
  std::string last_player, next_player;
  std::vector<std::vector<char> > board_data;
  int last_player_score = 0;
  int next_player_score = 0;

  pBoard = NULL;
  error.clear();

  if (!doc.HasMember("lastPlay")) {
    error = "AI first (no lastPlay found)";
    return ERROR_NO_LAST_PLAY;
  }

  if (!extractRequiredFields(doc, x, y, last_player, next_player, goal)) {
    error = "Missing required fields.";
    return ERROR_UNKNOWN;
  }

  *last_x = x;
  *last_y = y;

  if (!parseBoard(doc, board_data)) {
    error = "Invalid board field.";
    return ERROR_INVALID_BOARD;
  }

  if (!parseScores(doc, last_player, next_player, last_player_score, next_player_score)) {
    error = "Invalid scores field.";
    return ERROR_INVALID_SCORES;
  }

  pBoard =
      new Board(board_data, goal, last_player, next_player, last_player_score, next_player_score);

  std::cout << "Parsed Board State:\n" << std::endl;
  pBoard->printBitboard();

  // Obtain captured stones, if any.
  bool stoneCaptured =
      Rules::detectCaptureStones(*pBoard, x, y, (last_player == "X") ? PLAYER_1 : PLAYER_2);

  // If capture occurred, print and remove captured stones.
  if (stoneCaptured) {
    // std::cout << "Captured stones:" << std::endl;
    // for (std::vector<std::pair<int, int> >::iterator it = capturedStones.begin();
    //      it != capturedStones.end(); ++it) {
    //   std::cout << " - (" << it->first << ", " << it->second << ")" << std::endl;
    // }
    // std::cout << std::flush;
    pBoard->applyCapture();

    std::cout << "after" << std::endl;
    // pBoard->printBitboard();
    return PARSE_OK;
  }

  bool doubleThreeBit =
      Rules::detectDoublethreeBit(*pBoard, x, y, (last_player == "X") ? PLAYER_1 : PLAYER_2);
  if (doubleThreeBit) {
    error = "doublethree";
    delete pBoard;
    pBoard = NULL;
    return ERROR_DOUBLE_THREE;
  }

  return PARSE_OK;
}
