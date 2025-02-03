FROM debian:latest

RUN apt-get update && apt-get install -y \
    g++ cmake make libwebsockets-dev rapidjson-dev entr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

EXPOSE 8005

CMD find src inc -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "Makefile" \) | entr -n -r /bin/sh -c "make re && ./server"
