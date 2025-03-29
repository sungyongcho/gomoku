#include "server.hpp"

// Define WebSocket protocols (only WebSocket, no HTTP)
static struct lws_protocols protocols[] = {{"debug-protocol", callbackDebug, 0, 0, 0, NULL, 0},
                                           {NULL, NULL, 0, 0, 0, NULL, 0}};

Server::Server(int port) {
  memset(&info, 0, sizeof(info));
  info.port = port;
  info.protocols = protocols;
  info.options = LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;

  context = lws_create_context(&info);
  if (!context) {
    std::cerr << "Libwebsockets context creation failed!" << std::endl;
    exit(1);
  }
  // Minimax::initCombinedPatternScoreTablesToRemove();
  Evaluation::initCombinedPatternScoreTables();
  initZobrist();
  std::cout << "WebSocket Server running on ws://localhost:" << port << "/ws/debug" << std::endl;
}

void Server::run() {
  while (true) {
    lws_service(context, 100);
  }
  lws_context_destroy(context);
}
