#include "websocket_handler.hpp"

#include <cstring>

#include "json_parser.hpp"
#include "Minimax.hpp"
#include "request_handlers.hpp"
#include "response_builder.hpp"

int callbackWebsocket(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in,
                      size_t len) {
  psd_debug *psd = static_cast<psd_debug *>(user);  // <-- persistent!
  switch (reason) {
    case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION: {
      // only taking uri of /ws and /ws/debug
      char uri[128];

      if (lws_hdr_copy(wsi, uri, sizeof(uri), WSI_TOKEN_GET_URI) <= 0) {
        std::cerr << "Failed to get URI\n";
        return 1;
      }
      if (std::strcmp(uri, "/ws") != 0 && std::strcmp(uri, "/ws/debug") != 0) {
        std::cerr << "Rejected: invalid URI: " << uri << std::endl;
        return 1;  // Reject connection
      }

      new (&psd->difficulty) std::string();  // placement-new
      break;
    }

    case LWS_CALLBACK_ESTABLISHED:
      std::cout << "WebSocket `/ws` connected!" << std::endl;
      psd->difficulty.clear();  // starts empty for this client
      initZobrist();
      // testZobristHashingLogic();
      transTable.clear();
      break;
    case LWS_CALLBACK_RECEIVE: {
      std::string received_msg((char *)in, len);
      std::cout << "Received: " << received_msg << std::endl;

      rapidjson::Document doc;
      if (doc.Parse(received_msg.c_str()).HasParseError()) {
        sendErrorResponse(wsi, ERROR_UNKNOWN, "JSON Parse Error");
        return -1;
      }

      if (!doc.HasMember("type") || !doc["type"].IsString()) {
        sendErrorResponse(wsi, ERROR_UNKNOWN, "Invalid 'type' field");
        return -1;
      }

      std::string type = doc["type"].GetString();
      if (type == "move") {
        return handleMoveRequest(wsi, doc, psd);
      } else if (type == "evaluate") {
        return handleEvaluateRequest(wsi, doc);
      } else if (type == "test") {
        return handleTestRequest(wsi, doc);
      } else if (type == "ping") {
        sendJsonResponse(wsi, "{\"type\":\"pong\"}");
        return 0;
      } else if (type == "reset") {
        handleResetRequest(psd);
      } else {
        sendErrorResponse(wsi, ERROR_UNKNOWN, "Unknown type");
        return -1;
      }
      break;
    }

    case LWS_CALLBACK_CLOSED:
      psd->difficulty.~basic_string();  // manual dtor
      std::cout << "WebSocket connection closed." << std::endl;
      break;

    default:
      break;
  }
  return 0;
}
