#include <libwebsockets.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

// WebSocket callback function
static int callback_ws(struct lws *wsi, enum lws_callback_reasons reason,
                       void *user, void *in, size_t len) {
    // Fix unused parameters warnings
    (void)user;
    (void)in;
    (void)len;

    switch (reason) {
        case LWS_CALLBACK_ESTABLISHED:
            std::cout << "WebSocket connection established!" << std::endl;
            break;

        case LWS_CALLBACK_RECEIVE: {
            std::string received_msg((char*)in, len);
            std::cout << "Received: " << received_msg << std::endl;

            // Echo the message back
            lws_write(wsi, (unsigned char*)received_msg.c_str(), received_msg.length(), LWS_WRITE_TEXT);
            break;
        }

        case LWS_CALLBACK_CLOSED:
            std::cout << "WebSocket connection closed." << std::endl;
            break;

        default:
            break;
    }
    return 0;
}

// HTTP callback function
static int callback_http(struct lws *wsi, enum lws_callback_reasons reason,
                         void *user, void *in, size_t len) {
    // Fix unused parameters warnings
    (void)user;
    (void)in;
    (void)len;

    if (reason == LWS_CALLBACK_HTTP) {
        std::cout << "HTTP GET / request received!" << std::endl;

        // Create JSON response using RapidJSON
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
        doc.AddMember("message", "Hello world!", allocator);

        // Convert JSON to string
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        std::string json_str = buffer.GetString();

        // Send JSON response
        lws_write(wsi, (unsigned char*)json_str.c_str(), json_str.length(), LWS_WRITE_HTTP);
        return 0;
    }
    return 0;
}

// Main entry point
int main() {
    struct lws_protocols protocols[] = {
        { "http-only", callback_http, 0, 0, 0, NULL, 0 },
        { "websocket", callback_ws, 0, 0, 0, NULL, 0 },
        { NULL, NULL, 0, 0, 0, NULL, 0 } // Ensure all fields are initialized
    };

    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));
    info.port = 8005;
    info.protocols = protocols;
    info.options = LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;

    struct lws_context *context = lws_create_context(&info);
    if (!context) {
        std::cerr << "Libwebsockets context creation failed!" << std::endl;
        return -1;
    }

    std::cout << "Server running on ws://localhost:8005 and http://localhost:8005" << std::endl;

    while (true) {
        lws_service(context, 100);
    }

    lws_context_destroy(context);
    return 0;
}
