# Use a lightweight Debian-based image
FROM debian:latest

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    g++ cmake make libboost-all-dev nlohmann-json3-dev entr\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

EXPOSE 8005

CMD find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "Makefile" \) | entr -n -r /bin/sh -c "make && ./server"

