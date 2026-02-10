def _to_index(x: int, y: int, board_size: int) -> int:
    return y * board_size + x


def test_candidate_moves_empty_board(cpp_core) -> None:
    state = cpp_core.initial_state()
    candidates = cpp_core.get_candidate_moves(state.board)
    board_size = cpp_core.board_size
    center = board_size // 2
    expected_idx = _to_index(center, center, board_size)
    print("empty candidates size:", len(candidates), "values:", candidates)
    assert candidates == [expected_idx]


def test_candidate_moves_single_stone_corner(cpp_core) -> None:
    state = cpp_core.apply_move(cpp_core.initial_state(), 0, 0, 1)
    board_size = cpp_core.board_size
    candidates = cpp_core.get_candidate_moves(state.board)

    expected = {
        _to_index(x, y, board_size)
        for y in range(0, min(3, board_size))
        for x in range(0, min(3, board_size))
    }
    expected.discard(_to_index(0, 0, board_size))

    print("corner candidates size:", len(candidates), "values:", candidates)
    assert set(candidates) == expected


def test_candidate_moves_merge_without_duplicates(cpp_core) -> None:
    state = cpp_core.apply_move(cpp_core.initial_state(), 2, 2, 1)
    state = cpp_core.apply_move(state, 4, 2, 2)
    board_size = cpp_core.board_size
    board = state.board
    candidates = cpp_core.get_candidate_moves(board)

    expected = set()
    for x, y in [(2, 2), (4, 2)]:
        for dy in range(-2, 3):
            ny = y + dy
            if ny < 0 or ny >= board_size:
                continue
            for dx in range(-2, 3):
                nx = x + dx
                if dx == 0 and dy == 0:
                    continue
                if nx < 0 or nx >= board_size:
                    continue
                idx = _to_index(nx, ny, board_size)
                if board[idx] == 0:
                    expected.add(idx)

    print("merge candidates size:", len(candidates), "values (sample 10):", candidates[:10])
    assert set(candidates) == expected
