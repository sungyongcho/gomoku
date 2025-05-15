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

// ─── protocol table ────────────────────────────────────────────────
static struct lws_protocols protocols[] = {
    {"debug-protocol", callbackDebug,
     sizeof(psd_debug),  // ← allocate this many bytes per connection
     0,                  // rx buffer size (unused)
     0, NULL, 0},
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

  /* ── TLS only if both env-vars exist ─────────────────── */
  const char *cert = std::getenv("CERT_FULLCHAIN");
  const char *key = std::getenv("CERT_PRIVKEY");
  if (cert && *cert && key && *key) {
    info.options |= LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
    info.ssl_cert_filepath = cert;
    info.ssl_private_key_filepath = key;
    info.ssl_cipher_list =
        "ECDHE-ECDSA-AES256-GCM-SHA384:"
        "ECDHE-RSA-AES256-GCM-SHA384:"
        "ECDHE-ECDSA-AES128-GCM-SHA256:"
        "ECDHE-RSA-AES128-GCM-SHA256";
    std::cout << "[TLS] enabled on port " << port << std::endl;
  } else {
    std::cout << "[TLS] disabled (CERT_* env vars missing)" << std::endl;
  }

  context = lws_create_context(&info);
  if (!context) {
    throw std::runtime_error("Libwebsockets context creation failed!");
  }

  // Initialize evaluation tables.
  Evaluation::initCombinedPatternScoreTables();
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
