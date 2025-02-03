#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <iostream>
#include <string>
#include <nlohmann/json.hpp> // JSON handling
#include <csignal>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;

asio::io_context ioc;  // Global IO context
tcp::acceptor *global_acceptor = nullptr;  // Global acceptor for cleanup

// Handle termination signals (SIGINT/SIGTERM)
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Cleaning up..." << std::endl;
    if (global_acceptor) {
        global_acceptor->close();  // Close the acceptor to release the port
    }
    ioc.stop();  // Stop the IO context
    exit(0);  // Exit the program safely
}

// WebSocket session (Handles a single client)
void do_websocket_session(tcp::socket socket) {
    try {
        websocket::stream<tcp::socket> ws(std::move(socket));
        ws.accept();

        beast::flat_buffer buffer;
        while (true) {
            ws.read(buffer);
            std::string message = beast::buffers_to_string(buffer.data());
            std::cout << "Received: " << message << std::endl;

            // Echo message back
            ws.text(ws.got_text());
            ws.write(buffer.data());
            buffer.consume(buffer.size());
        }
    } catch (std::exception &e) {
        std::cerr << "WebSocket Error: " << e.what() << std::endl;
    }
}

// HTTP session (Handles an HTTP request)
void do_http_session(tcp::socket socket) {
    try {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        http::read(socket, buffer, req);

        // Prepare HTTP response
        http::response<http::string_body> res{http::status::ok, req.version()};
        res.set(http::field::content_type, "application/json");

        nlohmann::json response_json;
        response_json["message"] = "Hello world!";
        res.body() = response_json.dump();
        res.prepare_payload();

        http::write(socket, res);
        socket.shutdown(tcp::socket::shutdown_send);
    } catch (std::exception &e) {
        std::cerr << "HTTP Error: " << e.what() << std::endl;
    }
}

// Main Server
int main() {
    try {
        // Register signal handlers for SIGINT (Ctrl+C) and SIGTERM (Docker stop)
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        tcp::acceptor acceptor(ioc, {tcp::v4(), 8005});
        global_acceptor = &acceptor;  // Store acceptor globally for cleanup

        std::cout << "Server running on http://localhost:8004" << std::endl;

        while (true) {
            tcp::socket socket(ioc);
            acceptor.accept(socket);

            // Detect WebSocket Upgrade Request
            beast::flat_buffer buffer;
            http::request<http::string_body> req;
            http::read(socket, buffer, req);

            if (websocket::is_upgrade(req)) {
                std::cout << "WebSocket connection request detected!" << std::endl;
                std::thread(do_websocket_session, std::move(socket)).detach();
            } else {
                std::thread(do_http_session, std::move(socket)).detach();
            }
        }
    } catch (std::exception &e) {
        std::cerr << "Server Error: " << e.what() << std::endl;
    }

    return 0;
}
