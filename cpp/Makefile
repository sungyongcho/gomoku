CXX = g++
# Common flags
COMMON_CXXFLAGS = -std=c++98 -Wall -Wextra -Werror -Iinc -Iinc/gomoku -Iinc/ws
LDFLAGS = -lwebsockets -lpthread

# Release and debug specific flags
RELEASE_CXXFLAGS = $(COMMON_CXXFLAGS)
DEBUG_CXXFLAGS = $(COMMON_CXXFLAGS) -g

SRC_DIR = src
# Two separate build directories
BUILD_DIR = build
DEBUG_BUILD_DIR = build_debug

# Find all .cpp files (matches one, two, or three levels of directory depth)
SRC = $(wildcard $(SRC_DIR)/*.cpp) \
      $(wildcard $(SRC_DIR)/*/*.cpp) \
      $(wildcard $(SRC_DIR)/*/*/*.cpp)

# Object files for release and debug
OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC))
DEBUG_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(DEBUG_BUILD_DIR)/%.o, $(SRC))

TARGET = server
DEBUG_TARGET = server_debug

# Default target: release build
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(RELEASE_CXXFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

# Debug target builds an executable with debugging symbols
debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(DEBUG_OBJ)
	$(CXX) $(DEBUG_CXXFLAGS) -o $(DEBUG_TARGET) $(DEBUG_OBJ) $(LDFLAGS)

# Pattern rules for release objects (one level)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(RELEASE_CXXFLAGS) -c $< -o $@

# Two levels deep for release
$(BUILD_DIR)/%.o: $(SRC_DIR)/*/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(RELEASE_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/*/*/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(RELEASE_CXXFLAGS) -c $< -o $@

# Pattern rules for debug objects (one level)
$(DEBUG_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(DEBUG_BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@

# Two levels deep for debug
$(DEBUG_BUILD_DIR)/%.o: $(SRC_DIR)/*/%.cpp | $(DEBUG_BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@

$(DEBUG_BUILD_DIR)/%.o: $(SRC_DIR)/*/*/%.cpp | $(DEBUG_BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(DEBUG_CXXFLAGS) -c $< -o $@

# Ensure build directories exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DEBUG_BUILD_DIR):
	mkdir -p $(DEBUG_BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(DEBUG_BUILD_DIR) $(TARGET) $(DEBUG_TARGET)

re: clean all

re_debug: clean debug
