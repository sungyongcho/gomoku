# Compiler and flags
CXX        := g++
CXXFLAGS   := -std=c++98 -Wall -Wextra -Iinc -Iinc/gomoku -Iinc/ws -O2
LDFLAGS    := -lwebsockets -lpthread

# Directories
SRC_DIR    := src
BUILD_DIR  := build
DEBUG_DIR  := build_debug

# Find all .cpp under SRC_DIR (any depth)
SRCS       := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS       := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEBUG_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(DEBUG_DIR)/%.o,$(SRCS))

# Targets
TARGET         := server
DEBUG_TARGET   := server_debug



# Compile rule (re-used for both builds)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS)      -c $< -o $@

$(DEBUG_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -g   -c $< -o $@

# Link — only runs if any .o is newer than the executable
$(TARGET): $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

all: $(TARGET)

debug: $(DEBUG_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(DEBUG_DIR)

fclean: clean
	rm -f $(TARGET) $(DEBUG_TARGET)

re: fclean all

re_debug: fclean debug

.PHONY: all debug clean re re_debug
