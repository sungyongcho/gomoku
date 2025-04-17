FROM debian:latest

# Copy dependencies and the install script
COPY dependencies.txt /tmp/dependencies.txt
COPY install-dependencies.sh /tmp/install-dependencies.sh

# Make the script executable
RUN chmod +x /tmp/install-dependencies.sh

# Run the installation script
WORKDIR /tmp
RUN /tmp/install-dependencies.sh
RUN git clone https://github.com/eradman/entr && cd entr/ && ./configure && make test && make install

# Fix file watcher on Mac (see: https://github.com/eradman/entr/issues/3)
ENV ENTR_INOTIFY_WORKAROUND=1
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

EXPOSE 8005

# Initial build: only compiles what's out‑of‑date
RUN make

# On file changes, rebuild only what’s needed, then run
CMD find src inc -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "Makefile" \) \
    | entr -n -r sh -c "make && ./server"
