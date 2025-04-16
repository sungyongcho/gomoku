#include <csignal>

#include "server.hpp"

volatile std::sig_atomic_t stopFlag = 0;

void handleSignal(int signal) {
  (void)signal;  // Mark parameter as unused
  stopFlag = 1;
}

int main() {
  // Register signal handlers for graceful shutdown.
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  try {
    Server server(8005);
    server.run(stopFlag);
  } catch (const std::exception &ex) {
    std::cerr << "Server initialization failed: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}
