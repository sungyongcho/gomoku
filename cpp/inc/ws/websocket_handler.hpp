#ifndef WEBSOCKET_HANDLER_HPP
#define WEBSOCKET_HANDLER_HPP

#include <libwebsockets.h>
#include <iostream>
#include <string>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "Board.hpp"

int callback_debug(struct lws *wsi, enum lws_callback_reasons reason,
                   void *user, void *in, size_t len);

Board* parse_json(struct lws *wsi, const rapidjson::Document &doc);

void success_response(struct lws *wsi, Board &board);

#endif
