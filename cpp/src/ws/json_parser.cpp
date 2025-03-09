#include "json_parser.hpp"

void send_json_response(struct lws *wsi, const std::string &response)
{
	size_t resp_len = response.size();
	size_t buf_size = LWS_PRE + resp_len;
	unsigned char *buf = new unsigned char[buf_size];
	unsigned char *p = &buf[LWS_PRE];
	memcpy(p, response.c_str(), resp_len);
	lws_write(wsi, p, resp_len, LWS_WRITE_TEXT);
	delete[] buf;
}

bool extract_required_fields(const rapidjson::Document &doc, int &x, int &y,
							 std::string &last_player, std::string &next_player, int &goal)
{
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

std::vector<std::vector<char> > parse_board_from_json(const rapidjson::Document &doc)
{
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
	return board_data;
}

bool parse_board(const rapidjson::Document &doc, std::vector<std::vector<char> > &board_data)
{
	if (!doc.HasMember("board") || !doc["board"].IsArray())
	{
		std::cerr << "Error: Missing or invalid 'board' field." << std::endl;
		return false;
	}

	board_data = parse_board_from_json(doc);
	return true;
}

bool parse_scores(const rapidjson::Document &doc, int &last_player_score, int &next_player_score)
{
	if (!doc.HasMember("scores") || !doc["scores"].IsArray())
	{
		std::cerr << "Error: Missing or invalid 'scores' field." << std::endl;
		return false;
	}

	for (rapidjson::SizeType i = 0; i < doc["scores"].Size(); i++)
	{
		std::string player = doc["scores"][i]["player"].GetString();
		int score = doc["scores"][i]["score"].GetInt();
		if (player == "X")
			last_player_score = score;
		else if (player == "O")
			next_player_score = score;
	}

	return true;
}

// Main parse_json function using ParseResult enum
ParseResult parse_json(const rapidjson::Document &doc, Board *&pBoard, std::string &error)
{
	int x, y, goal;
	std::string last_player, next_player;
	std::vector<std::vector<char> > board_data;
	int last_player_score = 0;
	int next_player_score = 0;

	pBoard = NULL;
	error.clear();

	if (!doc.HasMember("lastPlay"))
	{
		error = "AI first (no lastPlay found)";
		return ERROR_NO_LAST_PLAY;
	}

	if (!extract_required_fields(doc, x, y, last_player, next_player, goal))
	{
		error = "Missing required fields.";
		return ERROR_UNKNOWN;
	}

	if (!parse_board(doc, board_data))
	{
		error = "Invalid board field.";
		return ERROR_INVALID_BOARD;
	}

	if (!parse_scores(doc, last_player_score, next_player_score))
	{
		error = "Invalid scores field.";
		return ERROR_INVALID_SCORES;
	}

	pBoard = new Board(board_data, goal, last_player, next_player,
					   last_player_score, next_player_score);

	std::cout << "Parsed Board State:\n"
			  << pBoard->convert_board_for_print() << std::endl;

	// Obtain captured stones, if any.
	std::vector<std::pair<int, int> > capturedStones;
	std::cout << "before" << std::endl;
	pBoard->print_board_bit();
	bool stoneCaptured = Rules::get_captured_stones_bit(*pBoard, x, y, last_player, capturedStones);

	// If capture occurred, print and remove captured stones.
	if (stoneCaptured)
	{
		std::cout << "Captured stones:" << std::endl;
		for (std::vector<std::pair<int, int> >::iterator it = capturedStones.begin();
			 it != capturedStones.end(); ++it)
		{
			std::cout << " - (" << it->first << ", " << it->second << ")" << std::endl;
		}
		std::cout << std::flush;

		std::cout << "after" << std::endl;
		pBoard->print_board_bit();
		return PARSE_OK;
	}

	bool doubleThreeBit = Rules::double_three_detected_bit(*pBoard, x, y,
													(last_player == "X") ? PLAYER_1 : PLAYER_2);
	if (doubleThreeBit)
	{
		std::cout << "testing" << std::endl;
		error = "doublethree";
		delete pBoard;
		pBoard = NULL;
		return ERROR_DOUBLE_THREE;
	}

	// Only check double three if no capture occurred.
	bool doubleThree = Rules::double_three_detected(*pBoard, x, y,
													(last_player == "X") ? PLAYER_1 : PLAYER_2);

	if (doubleThree)
	{
		error = "doublethree";
		delete pBoard;
		pBoard = NULL;
		return ERROR_DOUBLE_THREE;
	}

	return PARSE_OK;
}
