#ifndef SERVER_HPP
#define SERVER_HPP

#include <libwebsockets.h>

#include <cstring>
#include <iostream>

#include "minimax.hpp"
#include "websocket_handler.hpp"

class Server {
 public:
  Server(int port);
  void run();

 private:
  struct lws_context *context;
  struct lws_context_creation_info info;
};

#endif  // SERVER_HPP
