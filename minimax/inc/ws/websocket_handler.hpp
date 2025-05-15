#ifndef WEBSOCKET_HANDLER_HPP
#define WEBSOCKET_HANDLER_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Board.hpp"
#include "Evaluation.hpp"
#include "json_parser.hpp"
struct psd_debug {
  std::string difficulty;  // keep the last difficulty here
};

int callbackDebug(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in,
                  size_t len);

void responseSuccessMove(struct lws *wsi, Board &board);

std::string constructErrorResponse(ParseResult result, const std::string &details);

#endif  // WEBSOCKET_HANDLER_HPP
