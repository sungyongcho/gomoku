#ifndef WEBSOCKET_HANDLER_HPP
#define WEBSOCKET_HANDLER_HPP

#include <libwebsockets.h>

#include <string>

struct psd_debug {
  std::string difficulty;  // keep the last difficulty here
};

int callbackWebsocket(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in,
                      size_t len);

#endif  // WEBSOCKET_HANDLER_HPP
