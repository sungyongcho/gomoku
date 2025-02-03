#include "websocket_handler.hpp"

int callback_debug(struct lws *wsi, enum lws_callback_reasons reason,
                   void *user, void *in, size_t len) {
    (void)user;

    switch (reason) {
        case LWS_CALLBACK_ESTABLISHED:
            std::cout << "WebSocket `/ws/debug` connected!" << std::endl;
            break;

        case LWS_CALLBACK_RECEIVE: {
            std::string received_msg((char*)in, len);
            std::cout << "Received: " << received_msg << std::endl;

            // Parse JSON using RapidJSON
            rapidjson::Document doc;
            if (doc.Parse(received_msg.c_str()).HasParseError()) {
                std::cerr << "JSON Parse Error!" << std::endl;
                return 1;
            }

            // Convert parsed JSON back to string
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc.Accept(writer);
            std::string json_str = buffer.GetString();

            // Send JSON response back
            unsigned char *send_buf = new unsigned char[LWS_SEND_BUFFER_PRE_PADDING + json_str.length() + LWS_SEND_BUFFER_POST_PADDING];
            std::memcpy(send_buf + LWS_SEND_BUFFER_PRE_PADDING, json_str.c_str(), json_str.length());

            lws_write(wsi, send_buf + LWS_SEND_BUFFER_PRE_PADDING, json_str.length(), LWS_WRITE_TEXT);
            delete[] send_buf;
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
