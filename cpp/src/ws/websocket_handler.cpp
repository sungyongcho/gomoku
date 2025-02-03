#include "websocket_handler.hpp"
#include "board.hpp"

int parse_json(struct lws *wsi, const rapidjson::Document &doc)
{
    if (!doc.HasMember("lastPlay"))
    {
        std::cerr << "AI first (no lastPlay found)" << std::endl;

        // Send error response
        std::string response = "{\"type\":\"error\",\"status\":\"tba\"}";
        lws_write(wsi, (unsigned char *)response.c_str(), response.length(), LWS_WRITE_TEXT);
        return -1; // Indicate an error in parsing
    }

    // Extract fields
    int x = doc["lastPlay"]["coordinate"]["x"].GetInt();
    int y = doc["lastPlay"]["coordinate"]["y"].GetInt();
    std::string last_player = doc["lastPlay"]["stone"].GetString();
    std::string next_player = doc["nextPlayer"].GetString();
    int goal = doc["goal"].GetInt();

    std::cout << "Move received: \n";
    std::cout << "  Last Play: (" << x << ", " << y << ") by " << last_player << std::endl;
    std::cout << "  Next Player: " << next_player << std::endl;
    std::cout << "  Goal: " << goal << std::endl;

    // Handle board
    if (!doc.HasMember("board") || !doc["board"].IsArray())
    {
        std::cerr << "Error: Missing or invalid 'board' field." << std::endl;
        return -1;
    }

    // Convert board from JSON array to std::vector<std::vector<char> >
    std::vector<std::vector<char> > board_data;
    for (rapidjson::SizeType i = 0; i < doc["board"].Size(); i++)
    {
        std::vector<char> row;
        for (rapidjson::SizeType j = 0; j < doc["board"][i].Size(); j++)
        {
            row.push_back(doc["board"][i][j].GetString()[0]); // Convert JSON string to char
        }
        board_data.push_back(row);
    }

    // Handle scores
    if (!doc.HasMember("scores") || !doc["scores"].IsArray())
    {
        std::cerr << "Error: Missing or invalid 'scores' field." << std::endl;
        return -1;
    }

    int last_player_score = 0;
    int next_player_score = 0;
    for (rapidjson::SizeType i = 0; i < doc["scores"].Size(); i++)
    {
        std::string player = doc["scores"][i]["player"].GetString();
        int score = doc["scores"][i]["score"].GetInt();

        if (player == "X")
            last_player_score = score;
        else if (player == "O")
            next_player_score = score;
    }

    Board board(board_data, goal, last_player, next_player, last_player_score, next_player_score);

    std::cout << "Parsed Board State:\n" << board.convert_board_for_print() << std::endl;

    return 0; // Indicate success
}

// WebSocket callback function
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

        // Parse JSON using RapidJSON
        rapidjson::Document doc;
        if (doc.Parse(received_msg.c_str()).HasParseError())
        {
            std::cerr << "JSON Parse Error!" << std::endl;
            return -1;
        }

        // Validate required fields
        if (!doc.HasMember("type") || !doc["type"].IsString())
        {
            std::cerr << "Error: Missing or invalid 'type' field." << std::endl;
            return -1;
        }

        std::string type = doc["type"].GetString();
        if (type == "move")
        {
            return parse_json(wsi, doc);
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
