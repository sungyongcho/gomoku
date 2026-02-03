# Compiler and flags
CXX        := g++
CXXFLAGS   := -std=c++98 -Wall -Wextra -Werror -Wreorder -Iinc -Iinc/gomoku -Iinc/ws -O2 -DBOOST_DISABLE_PRAGMA_MESSAGE
DBGFLAGS   := -std=c++98 -Wall -Wextra -Werror -Wreorder -Iinc -Iinc/gomoku -Iinc/ws -g -DBOOST_DISABLE_PRAGMA_MESSAGE
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
TARGET         := minimax
DEBUG_TARGET   := minimax_debug

# Test Config
TEST_SRC       := doublethree_test.cpp
TEST_TARGET    := doublethree_test
# Exclude main.o and anything in ws/ directory from test linkage
TEST_OBJS      := $(filter-out $(BUILD_DIR)/main.o $(BUILD_DIR)/ws/%, $(OBJS))

# Standard Build Rules

all: $(TARGET)

debug: $(DEBUG_TARGET)

# Link Main Executable
$(TARGET): $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Test Target Rule
doublethree: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) $(TEST_OBJS) -o $@

# Compile rule (re-used for both builds)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS)      -c $< -o $@

$(DEBUG_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(DBGFLAGS)   -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(DEBUG_DIR)

fclean: clean
	rm -f $(TARGET) $(DEBUG_TARGET) $(TEST_TARGET)

re: fclean all

re_debug: fclean debug

.PHONY: all debug clean re re_debug doublethree
