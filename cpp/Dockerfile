# Use a lightweight Debian-based image
FROM debian:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    g++ cmake make libwebsockets-dev rapidjson-dev entr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Expose WebSocket & HTTP Port
EXPOSE 8005

# Build and run the server with hot reload
CMD find src -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "Makefile" \) | entr -n -r /bin/sh -c "make re && ./server"
