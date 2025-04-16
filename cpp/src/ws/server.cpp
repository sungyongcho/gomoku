#include "server.hpp"

// Include headers for port checking
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

// Helper function to check if a port is available.
static bool isPortAvailable(int port) {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    return false;  // Unable to create socket; assume port not available.
  }
  int opt = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const void *)&opt, sizeof(opt));

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;

  // Try binding to the port.
  int bindResult = bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));
  close(sockfd);
  return (bindResult == 0);
}

// Define WebSocket protocols (only WebSocket, no HTTP)
static struct lws_protocols protocols[] = {{"debug-protocol", callbackDebug, 0, 0, 0, NULL, 0},
                                           {NULL, NULL, 0, 0, 0, NULL, 0}};

Server::Server(int port) {
  // Check if the port is available before proceeding.
  if (!isPortAvailable(port)) {
    std::ostringstream oss;
    oss << "Port " << port << " is in use. Please free it or use a different port.";
    throw std::runtime_error(oss.str());
  }

  std::memset(&info, 0, sizeof(info));
  info.port = port;
  info.protocols = protocols;
  info.options = LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;

  context = lws_create_context(&info);
  if (!context) {
    throw std::runtime_error("Libwebsockets context creation failed!");
  }

  // Initialize evaluation tables.
  Evaluation::initCombinedPatternScoreTablesHard();

  std::cout << "WebSocket Server running on ws://localhost:" << port << "/ws/debug" << std::endl;
}

void Server::run(volatile std::sig_atomic_t &stopFlag) {
  while (!stopFlag) {
    lws_service(context, 100);
  }
  lws_context_destroy(context);
  std::cout << "Server shut down gracefully." << std::endl;
}
