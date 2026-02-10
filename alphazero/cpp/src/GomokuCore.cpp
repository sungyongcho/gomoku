#include "GomokuCore.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace {
constexpr std::array<std::pair<int, int>, 8> kDirections{{
    {0, -1},   // N
    {1, -1},   // NE
    {1, 0},    // E
    {1, 1},    // SE
    {0, 1},    // S
    {-1, 1},   // SW
    {-1, 0},   // W
    {-1, -1},  // NW
}};

constexpr std::array<std::pair<int, int>, 4> kUniqueDirections{{
    {0, -1},   // N
    {1, -1},   // NE
    {1, 0},    // E
    {1, 1},    // SE
}};

constexpr int kCandidateRadius = 2;
}  // namespace

// Static member definitions for C++14 (no inline variables)
constexpr int8_t GomokuCore::kEmpty;
constexpr int8_t GomokuCore::kPlayer1;
constexpr int8_t GomokuCore::kPlayer2;

GomokuCore::GomokuCore(
    int board_size,
    bool enable_doublethree,
    bool enable_capture,
    int capture_goal,
    int gomoku_goal,
    int history_length
)
    : board_size_(board_size),
      action_size_(board_size * board_size),
      gomoku_goal_(gomoku_goal),
      capture_goal_(capture_goal),
      history_length_(history_length),
      enable_capture_(enable_capture),
      enable_doublethree_(enable_doublethree) {}

GomokuState GomokuCore::initial_state() const {
    GomokuState st{};
    st.board.assign(action_size_, kEmpty);
    st.p1_pts = 0;
    st.p2_pts = 0;
    st.next_player = kPlayer1;
    st.last_move_idx = static_cast<int16_t>(-1);
    st.empty_count = static_cast<int16_t>(action_size_);
    st.history.clear();
    return st;
}

int GomokuCore::opponent(int player) {
    return player == kPlayer1 ? kPlayer2 : kPlayer1;
}

std::vector<int16_t> GomokuCore::detect_captures(
    const std::vector<int8_t>& board,
    int x,
    int y,
    int player
) const {
    std::vector<int16_t> captured;
    captured.reserve(16);

    const int opp = opponent(player);

    for (const auto& dir : kDirections) {
        const int dx = dir.first;
        const int dy = dir.second;
        const int x1 = x + dx;
        const int y1 = y + dy;
        const int x2 = x + 2 * dx;
        const int y2 = y + 2 * dy;
        const int x3 = x + 3 * dx;
        const int y3 = y + 3 * dy;

        if (x3 < 0 || x3 >= board_size_ || y3 < 0 || y3 >= board_size_) {
            continue;
        }

        if (get_pos(board, x1, y1) == opp && get_pos(board, x2, y2) == opp
            && get_pos(board, x3, y3) == player) {
            captured.push_back(static_cast<int16_t>(flat_index(x1, y1)));
            captured.push_back(static_cast<int16_t>(flat_index(x2, y2)));
        }
    }

    return captured;
}

bool GomokuCore::check_local_gomoku(
    const std::vector<int8_t>& board,
    int x,
    int y,
    int player
) const {
    for (const auto& dir : kUniqueDirections) {
        const int dx = dir.first;
        const int dy = dir.second;
        int count = 1;

        int nx = x + dx;
        int ny = y + dy;
        while (
            nx >= 0 && nx < board_size_ && ny >= 0 && ny < board_size_
            && get_pos(board, nx, ny) == player
        ) {
            ++count;
            if (count >= gomoku_goal_) {
                return true;
            }
            nx += dx;
            ny += dy;
        }

        nx = x - dx;
        ny = y - dy;
        while (
            nx >= 0 && nx < board_size_ && ny >= 0 && ny < board_size_
            && get_pos(board, nx, ny) == player
        ) {
            ++count;
            if (count >= gomoku_goal_) {
                return true;
            }
            nx -= dx;
            ny -= dy;
        }
    }
    return false;
}

void GomokuCore::populate_forbidden_finder(
    CForbiddenPointFinder& finder,
    const std::vector<int8_t>& board,
    int player
) const {
    finder.Clear();
    if (finder.boardSize_ != board_size_) {
        finder.ResizeBoard(board_size_);
    }
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const int8_t v = get_pos(board, x, y);
            if (v == kEmpty) {
                continue;
            }
            int8_t mapped = v;
            if (player == kPlayer2) {
                if (v == kPlayer1) {
                    mapped = kPlayer2;
                } else if (v == kPlayer2) {
                    mapped = kPlayer1;
                }
            }
            finder.SetStone(x, y, static_cast<char>(mapped));
        }
    }
}

bool GomokuCore::is_double_three(
    CForbiddenPointFinder& finder,
    int x,
    int y,
    int player
) const {
    if (!enable_doublethree_) {
        return false;
    }
    (void)player;
    return finder.IsDoubleThree(x, y);
}

GomokuState GomokuCore::apply_move(
    const GomokuState& state,
    int x,
    int y,
    int player
) const {
    GomokuState next = state;
    if (next.board.size() != static_cast<size_t>(action_size_)) {
        next.board.assign(action_size_, kEmpty);
    }

    set_pos(next.board, x, y, static_cast<int8_t>(player));
    next.last_move_idx = static_cast<int16_t>(flat_index(x, y));
    next.empty_count = static_cast<int16_t>(next.empty_count - 1);

    if (enable_capture_) {
        const auto captured = detect_captures(next.board, x, y, player);
        if (!captured.empty()) {
            for (const auto idx : captured) {
                set_pos(next.board, idx % board_size_, idx / board_size_, kEmpty);
            }
            const int captured_pairs = static_cast<int>(captured.size() / 2);
            if (player == kPlayer1) {
                next.p1_pts = static_cast<int16_t>(next.p1_pts + captured_pairs);
            } else {
                next.p2_pts = static_cast<int16_t>(next.p2_pts + captured_pairs);
            }
            next.empty_count = static_cast<int16_t>(next.empty_count + captured.size());
        }
    }

    next.next_player = static_cast<int8_t>(opponent(player));

    if (history_length_ > 0) {
        std::vector<int16_t> new_history;
        new_history.reserve(static_cast<size_t>(history_length_));
        new_history.push_back(next.last_move_idx);
        for (size_t i = 0; i < state.history.size() && i + 1 < static_cast<size_t>(history_length_); ++i) {
            new_history.push_back(state.history[i]);
        }
        next.history = std::move(new_history);
    } else {
        next.history.clear();
    }

    return next;
}

bool GomokuCore::check_win(const GomokuState& state, int x, int y) const {
    const int8_t player = get_pos(state.board, x, y);
    if (check_local_gomoku(state.board, x, y, player)) {
        return true;
    }
    if (capture_goal_ > 0) {
        if (player == kPlayer1 && state.p1_pts >= capture_goal_) {
            return true;
        }
        if (player == kPlayer2 && state.p2_pts >= capture_goal_) {
            return true;
        }
    }
    return false;
}

std::vector<int16_t> GomokuCore::get_legal_moves(const GomokuState& state) const {
    std::vector<int16_t> legal;
    legal.reserve(static_cast<size_t>(state.empty_count));

    const int player = state.next_player;
    const int player_stones = static_cast<int>(
        std::count(state.board.begin(), state.board.end(), static_cast<int8_t>(player))
    );

    const bool should_check_doublethree = enable_doublethree_ && player_stones >= 4;
    CForbiddenPointFinder finder(board_size_);
    if (should_check_doublethree) {
        populate_forbidden_finder(finder, state.board, player);
    }

    for (int idx = 0; idx < action_size_; ++idx) {
        if (state.board[idx] != kEmpty) {
            continue;
        }
        const int x = idx % board_size_;
        const int y = idx / board_size_;

        if (should_check_doublethree && is_double_three(finder, x, y, player)) {
            continue;
        }
        legal.push_back(static_cast<int16_t>(idx));
    }
    return legal;
}

std::vector<int> GomokuCore::GetCandidateMoves(const std::vector<int>& board_state) const {
    if (board_state.size() != static_cast<size_t>(action_size_)) {
        throw std::invalid_argument("board_state size mismatch");
    }

    std::vector<int> candidates;
    candidates.reserve(64);
    std::vector<uint8_t> visited(static_cast<size_t>(action_size_), 0);
    bool is_empty = true;

    for (int y = 0; y < board_size_; ++y) {
        const int row_offset = y * board_size_;
        for (int x = 0; x < board_size_; ++x) {
            const int idx = row_offset + x;
            if (board_state[idx] == 0) {
                continue;
            }
            is_empty = false;

            for (int dy = -kCandidateRadius; dy <= kCandidateRadius; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= board_size_) {
                    continue;
                }
                const int ny_offset = ny * board_size_;
                for (int dx = -kCandidateRadius; dx <= kCandidateRadius; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int nx = x + dx;
                    if (!is_on_board(nx, ny)) {
                        continue;
                    }
                    const int n_idx = ny_offset + nx;
                    if (board_state[n_idx] == 0 && visited[n_idx] == 0) {
                        visited[n_idx] = 1;
                        candidates.push_back(n_idx);
                    }
                }
            }
        }
    }

    if (is_empty) {
        if ((board_size_ & 1) == 0) {
            throw std::logic_error("board_size must be odd to return center on empty board");
        }
        const int center = board_size_ / 2;
        candidates.push_back(center * board_size_ + center);
    }

    return candidates;
}

void GomokuCore::write_state_features(const GomokuState& state, float* out) const {
    if (out == nullptr) {
        throw std::invalid_argument("output buffer is null");
    }

    const int base_channels = 8;
    const int total_channels = base_channels + history_length_;
    const int plane = board_size_ * board_size_;
    const size_t total = static_cast<size_t>(total_channels * plane);

    std::fill(out, out + total, 0.0f);

    const int player = state.next_player;
    const int opp = opponent(player);

    const auto plane_offset = [plane](int channel, int idx) -> size_t {
        return channel * plane + idx;
    };

    for (int idx = 0; idx < action_size_; ++idx) {
        const int8_t v = state.board[idx];
        if (v == player) {
            out[plane_offset(0, idx)] = 1.0f;
        } else if (v == opp) {
            out[plane_offset(1, idx)] = 1.0f;
        } else if (v == kEmpty) {
            out[plane_offset(2, idx)] = 1.0f;
        }
    }

    if (state.last_move_idx >= 0 && state.last_move_idx < action_size_) {
        out[plane_offset(3, state.last_move_idx)] = 1.0f;
    }

    if (enable_capture_ && capture_goal_ > 0) {
        const float my_pts = player == kPlayer1
            ? static_cast<float>(state.p1_pts)
            : static_cast<float>(state.p2_pts);
        const float opp_pts = player == kPlayer1
            ? static_cast<float>(state.p2_pts)
            : static_cast<float>(state.p1_pts);

        const float my_ratio = std::max(0.0f, std::min(1.0f, my_pts / static_cast<float>(capture_goal_)));
        const float opp_ratio = std::max(0.0f, std::min(1.0f, opp_pts / static_cast<float>(capture_goal_)));

        for (int idx = 0; idx < plane; ++idx) {
            out[plane_offset(4, idx)] = my_ratio;
            out[plane_offset(5, idx)] = opp_ratio;
        }
    }

    const float color_val = player == kPlayer1 ? 1.0f : -1.0f;
    for (int idx = 0; idx < plane; ++idx) {
        out[plane_offset(6, idx)] = color_val;
    }

    if (enable_doublethree_) {
        CForbiddenPointFinder finder(board_size_);
        populate_forbidden_finder(finder, state.board, player);

        for (int idx = 0; idx < action_size_; ++idx) {
            if (state.board[idx] != kEmpty) {
                continue;
            }
            const int x = idx % board_size_;
            const int y = idx / board_size_;
            if (is_double_three(finder, x, y, player)) {
                out[plane_offset(7, idx)] = 1.0f;
            }
        }
    }

    if (history_length_ > 0) {
        for (int k = 0; k < history_length_; ++k) {
            if (k >= static_cast<int>(state.history.size())) {
                break;
            }
            const int16_t move_idx = state.history[k];
            if (move_idx < 0 || move_idx >= action_size_) {
                continue;
            }
            const int channel = base_channels + k;
            out[plane_offset(channel, move_idx)] = 1.0f;
        }
    }
}

std::vector<float> GomokuCore::encode_state(const GomokuState& state) const {
    const int total_channels = 8 + history_length_;
    const int plane = board_size_ * board_size_;
    std::vector<float> features(static_cast<size_t>(total_channels * plane));
    write_state_features(state, features.data());
    return features;
}
