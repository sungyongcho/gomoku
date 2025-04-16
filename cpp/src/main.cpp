#include <csignal>

#include "server.hpp"

volatile std::sig_atomic_t stopFlag = 0;

void handleSignal(int signal) {
  (void)signal;  // Mark signal as unused
  stopFlag = 1;
}

int main() {
  // Register signal handlers for graceful shutdown
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  Server server(8005);
  server.run(stopFlag);  // Run the server with a stop flag
  return 0;
}
