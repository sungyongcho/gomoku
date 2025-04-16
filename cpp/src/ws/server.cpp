#include "server.hpp"

// Define WebSocket protocols (only WebSocket, no HTTP)
static struct lws_protocols protocols[] = {{"debug-protocol", callbackDebug, 0, 0, 0, NULL, 0},
                                           {NULL, NULL, 0, 0, 0, NULL, 0}};

Server::Server(int port) {
  std::memset(&info, 0, sizeof(info));
  info.port = port;
  info.protocols = protocols;
  info.options = LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;

  context = lws_create_context(&info);
  if (!context) {
    std::cerr << "Libwebsockets context creation failed!" << std::endl;
    exit(1);
  }
  // Evaluation initialization (if needed)
  Evaluation::initCombinedPatternScoreTablesHard();
  std::cout << "WebSocket Server running on ws://localhost:" << port << "/ws/debug" << std::endl;
}

void Server::run(volatile std::sig_atomic_t &stopFlag) {
  while (!stopFlag) {
    lws_service(context, 100);
  }
  // Once stopFlag is set, exit the loop and clean up
  lws_context_destroy(context);
  std::cout << "Server shut down gracefully." << std::endl;
}
