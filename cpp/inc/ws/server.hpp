#ifndef SERVER_HPP
#define SERVER_HPP

#include <libwebsockets.h>

#include <csignal>
#include <cstring>
#include <iostream>

#include "Evaluation.hpp"
#include "Gomoku.hpp"
#include "Minimax.hpp"
#include "websocket_handler.hpp"

class Server {
 public:
  Server(int port);
  void run(volatile std::sig_atomic_t &stopFlag);

 private:
  struct lws_context *context;
  struct lws_context_creation_info info;
};

#endif  // SERVER_HPP
