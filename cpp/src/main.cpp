#include "server.hpp"

int main() {
    Server server(8005);
    server.run();
    return 0;
}
