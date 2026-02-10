#pragma once

#include <cstdint>
#include <vector>

#include "ForbiddenPointFinder.h"

struct GomokuState {
    std::vector<int8_t> board;
    int16_t p1_pts;
    int16_t p2_pts;
    int8_t next_player;
    int16_t last_move_idx;
    int16_t empty_count;
    std::vector<int16_t> history;
};

class GomokuCore {
public:
    GomokuCore(
        int board_size,
        bool enable_doublethree,
        bool enable_capture,
        int capture_goal,
        int gomoku_goal,
        int history_length
    );

    GomokuState initial_state() const;

    GomokuState apply_move(const GomokuState& state, int x, int y, int player) const;

    bool check_win(const GomokuState& state, int x, int y) const;

    std::vector<int16_t> get_legal_moves(const GomokuState& state) const;

    std::vector<int> GetCandidateMoves(const std::vector<int>& board_state) const;

    std::vector<float> encode_state(const GomokuState& state) const;
    void write_state_features(const GomokuState& state, float* out) const;

    int board_size() const { return board_size_; }
    int action_size() const { return action_size_; }
    int history_length() const { return history_length_; }

private:
    static constexpr int8_t kEmpty = 0;
    static constexpr int8_t kPlayer1 = 1;
    static constexpr int8_t kPlayer2 = 2;

    int board_size_;
    int action_size_;
    int gomoku_goal_;
    int capture_goal_;
    int history_length_;
    bool enable_capture_;
    bool enable_doublethree_;

    inline int flat_index(int x, int y) const { return x + y * board_size_; }
    inline int8_t get_pos(const std::vector<int8_t>& board, int x, int y) const {
        return board[flat_index(x, y)];
    }
    inline void set_pos(std::vector<int8_t>& board, int x, int y, int8_t value) const {
        board[flat_index(x, y)] = value;
    }
    inline bool is_on_board(int x, int y) const {
        return x >= 0 && x < board_size_ && y >= 0 && y < board_size_;
    }
    static int opponent(int player);

    std::vector<int16_t> detect_captures(
        const std::vector<int8_t>& board,
        int x,
        int y,
        int player
    ) const;
    bool check_local_gomoku(
        const std::vector<int8_t>& board,
        int x,
        int y,
        int player
    ) const;
    bool is_double_three(
        CForbiddenPointFinder& finder,
        int x,
        int y,
        int player
    ) const;
    void populate_forbidden_finder(
        CForbiddenPointFinder& finder,
        const std::vector<int8_t>& board,
        int player
    ) const;
};
