#ifndef SERVER_HPP
#define SERVER_HPP

#include <libwebsockets.h>
#include <iostream>
#include <cstring>
#include "websocket_handler.hpp"

class Server {
public:
    Server(int port);
    void run();

private:
    struct lws_context *context;
    struct lws_context_creation_info info;
};

#endif // SERVER_HPP
