CXX = g++
CXXFLAGS = -std=c++98 -Wall -Wextra -Werror -Iinc -Iinc/gomoku -Iinc/ws
LDFLAGS = -lwebsockets -lpthread

SRC_DIR = src
BUILD_DIR = build
INC_DIR = inc

# Find all .cpp files in src/ and its subdirectories
SRC = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*/*.cpp) $(wildcard $(SRC_DIR)/*/*/*.cpp)
OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC))

TARGET = server

all: $(TARGET)

# Link all compiled object files to create the executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

# Ensure subdirectories exist before compiling
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)/$(dir %)
	@mkdir -p $(dir $@)   # Create subdirectory if it doesn't exist
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/*/%.cpp | $(BUILD_DIR)/$(dir %)
	@mkdir -p $(dir $@)   # Create subdirectory if it doesn't exist
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/*/*/%.cpp | $(BUILD_DIR)/$(dir %)
	@mkdir -p $(dir $@)   # Create subdirectory if it doesn't exist
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create subdirectories for object files
$(BUILD_DIR)/%/:
	mkdir -p $@

# Ensure the base build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

re: clean all
