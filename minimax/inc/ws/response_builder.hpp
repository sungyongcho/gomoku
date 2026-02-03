#ifndef RESPONSE_BUILDER_HPP
#define RESPONSE_BUILDER_HPP

#include <libwebsockets.h>

#include <string>

#include "Board.hpp"
#include "json_parser.hpp"

void responseSuccessMove(struct lws *wsi, Board &board, int aiPlayX, int aiPlayY,
                         double executionTime);
void responseSuccessEvaluate(struct lws *wsi, int evalScoreX, int evalScoreY);
std::string constructErrorResponse(ParseResult result, const std::string &details);
void sendErrorResponse(struct lws *wsi, ParseResult result, const std::string &details);

#endif  // RESPONSE_BUILDER_HPP
