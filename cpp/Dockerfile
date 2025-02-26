FROM debian:latest

# Copy dependencies and the install script
COPY dependencies.txt /tmp/dependencies.txt
COPY install-dependencies.sh /tmp/install-dependencies.sh

# Make the script executable
RUN chmod +x /tmp/install-dependencies.sh

# Run the installation script
RUN /tmp/install-dependencies.sh && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

EXPOSE 8005

# Compile the project before runtime for faster startup
RUN make re

CMD find src inc -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "Makefile" \) | \
    entr -n -r /bin/sh -c "./server"
