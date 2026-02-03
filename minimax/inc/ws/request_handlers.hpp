#ifndef REQUEST_HANDLERS_HPP
#define REQUEST_HANDLERS_HPP

#include <libwebsockets.h>
#include <rapidjson/document.h>

#include "websocket_handler.hpp"

int handleMoveRequest(struct lws *wsi, const rapidjson::Document &doc, psd_debug *psd);
int handleEvaluateRequest(struct lws *wsi, const rapidjson::Document &doc);
int handleTestRequest(struct lws *wsi, const rapidjson::Document &doc);
void handleResetRequest(psd_debug *psd);

#endif  // REQUEST_HANDLERS_HPP
